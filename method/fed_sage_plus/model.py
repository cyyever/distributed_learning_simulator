from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

predicated_missing_neighbor_num = 5
num_latent_feature = 128


class MendGraph(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        degree: torch.Tensor,
        generated_features: torch.Tensor,
    ) -> Any:
        new_edges = []
        generated_features = generated_features.view(
            -1, predicated_missing_neighbor_num, self.num_features
        )
        degree_list: list = degree.view(-1).tolist()
        node_len = x.shape[0]
        new_node_len = node_len
        new_features = []
        assert node_len > 0
        feature_dict: dict[int, dict[int, torch.Tensor]] = {}

        for i in range(node_len):
            feature_dict[i] = {}
            for j in range(
                min(predicated_missing_neighbor_num, max(0, int(degree_list[i])))
            ):
                new_features.append(generated_features[i][j])
                feature_dict[i][j] = new_features[-1]
                pair = [i, new_node_len]
                new_edges.append(pair)
                new_edges.append([pair[1], pair[0]])
                new_node_len += 1

        if not new_edges:
            return x, edge_index, {}
        generated_features = torch.vstack(new_features)
        mend_features = torch.cat((x, generated_features))
        mend_edge_index = torch.cat(
            (
                edge_index,
                torch.tensor(
                    new_edges, dtype=torch.long, device=edge_index.device
                ).transpose(0, 1),
            ),
            dim=1,
        )
        return (mend_features, mend_edge_index, feature_dict)


class RegModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reg_1 = nn.Linear(num_latent_feature, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.reg_1(x))
        return x


class Gen(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features

        self.fc1 = nn.Linear(num_latent_feature, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc_flat = nn.Linear(
            1024, predicated_missing_neighbor_num * self.num_features
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            rand = torch.normal(0, 1, size=x.shape, device=x.device)
            x = x + rand
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc_flat(x))
        return x


class LocalSagePlus(nn.Module):
    def __init__(
        self,
        graph_model_cls: Callable,
        num_features: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.encoder_model = graph_model_cls(
            num_features=num_features,
            num_classes=num_latent_feature,
        )
        self.reg_model = RegModel()
        self.gen = Gen(
            num_features=num_features,
        )
        self.mend_graph = MendGraph(
            num_features=num_features,
        )
        self.classifier = graph_model_cls(
            num_features=num_features,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs: Any) -> dict:
        new_feature = self.encoder_model(x, edge_index)
        degree = self.reg_model(new_feature)
        generated_features = self.gen(new_feature)
        mend_x, mend_edge_index, feature_dict = self.mend_graph(
            x=x,
            edge_index=edge_index,
            degree=degree,
            generated_features=generated_features,
        )
        pred_output = self.classifier(mend_x, mend_edge_index)
        return {
            "degree": degree,
            "generated_features": feature_dict,
            "output": pred_output,
        }
