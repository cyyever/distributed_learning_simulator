from functools import cached_property
from typing import Any, Callable

import torch
import torch.nn
import torch.nn.functional as F
from cyy_naive_lib.log import log_debug, log_error
from cyy_torch_graph import GraphModelEvaluator
from cyy_torch_toolbox import (ClassificationDatasetCollection, Executor,
                               MachineLearningPhase)
from torch_geometric.utils import degree as degree_fun
from torch_geometric.utils import k_hop_subgraph

from .model import LocalSagePlus


class FedSagePlusModelEvaluator(GraphModelEvaluator):
    edge_index_fun: Callable | None = None
    masked_edge_index_fun: Callable | None = None
    masked_node_list_fun: Callable | None = None

    def __compute_degree_loss(
        self, batch_mask: torch.Tensor, predicated_degree: torch.Tensor
    ) -> torch.Tensor:
        assert self.masked_edge_index_fun is not None
        masked_edge_index = self.masked_edge_index_fun()
        assert self.edge_index_fun is not None
        original_degree = degree_fun(index=self.edge_index_fun()[0])
        masked_degree = degree_fun(
            index=masked_edge_index[0], num_nodes=original_degree.shape[0]
        )
        degree = predicated_degree.view(-1)
        degree_diff = torch.zeros_like(degree)
        batch_mask_list = batch_mask.tolist()
        for idx, n_id in enumerate(self.n_id.tolist()):
            if not batch_mask_list[idx]:
                continue
            if n_id >= original_degree.shape[0]:
                assert n_id >= masked_degree.shape[0]
                continue
            degree_diff[idx] = original_degree[n_id] - masked_degree[n_id]
        return F.smooth_l1_loss(degree_diff[batch_mask], degree[batch_mask])

    @cached_property
    def old_features(self) -> torch.Tensor:
        return (
            self.get_dataset_util(phase=MachineLearningPhase.Training)
            .get_original_graph(0)
            .x
        )

    def __get_hidden_neighbor(self, node_idx: torch.Tensor) -> dict:
        edge_index_list: list
        assert self.edge_index_fun is not None
        edge_index = self.edge_index_fun()
        max_node = int(edge_index.view(-1).max().item())
        narrowed_edge_index = k_hop_subgraph(
            node_idx,
            num_hops=1,
            edge_index=edge_index,
            num_nodes=max(int(node_idx.view(-1).max().item()), max_node) + 1,
            relabel_nodes=False,
            flow="source_to_target",
        )[1]
        assert narrowed_edge_index.shape[1]
        assert self.masked_node_list_fun is not None
        masked_node_list = self.masked_node_list_fun()
        narrowed_edge_index = k_hop_subgraph(
            masked_node_list,
            num_hops=1,
            edge_index=narrowed_edge_index,
            num_nodes=max(max(masked_node_list), max_node) + 1,
            relabel_nodes=False,
            flow="target_to_source",
        )[1]
        edge_index_list = narrowed_edge_index.tolist()
        edge_dict: dict = {}

        for idx, src in enumerate(edge_index_list[0]):
            if src not in edge_dict:
                edge_dict[src] = set()
            edge_dict[src].add(edge_index_list[1][idx])
        assert edge_dict
        return edge_dict

    def _forward_model(self, inputs: Any, **kwargs: Any) -> dict:
        fun: Callable = self._get_forward_fun()
        output = fun(**inputs, batch_mask=kwargs["batch_mask"])
        return self._compute_loss(output=output, **kwargs)

    def __compute_feature_loss(
        self,
        batch_mask: torch.Tensor,
        generated_features: dict,
    ) -> torch.Tensor | None:
        assert self.masked_edge_index_fun is not None
        batch_mask_list = batch_mask.tolist()
        l2_sum: None | torch.Tensor = None
        n_id_list: list = self.n_id.tolist()
        node_idx = self.n_id[batch_mask]

        missed_neighbor = self.__get_hidden_neighbor(node_idx=node_idx)
        log_debug("generated_features len %s", len(generated_features))
        for idx in sorted(generated_features.keys()):
            assert batch_mask_list[idx]
            hidden_neighbor_set = missed_neighbor.get(n_id_list[idx], set())
            if not hidden_neighbor_set:
                continue
            for filled_feature in generated_features[idx].values():
                min_diff = None
                for hidden_neighbor in hidden_neighbor_set:
                    diff = torch.nn.MSELoss(reduction="sum")(
                        self.old_features[hidden_neighbor].to(
                            device=filled_feature.device, non_blocking=True
                        ),
                        filled_feature,
                    )
                    if min_diff is None or diff < min_diff:
                        min_diff = diff
                assert min_diff is not None
                if l2_sum is None:
                    l2_sum = min_diff
                else:
                    l2_sum = l2_sum + min_diff
        if l2_sum is not None:
            l2_sum = l2_sum / len([a for a in batch_mask_list if a])
        return l2_sum

    def _compute_loss(self, **kwargs: Any) -> dict:
        batch_mask = kwargs["batch_mask"]
        output = kwargs["output"]["output"]
        predicated_degree = kwargs["output"].pop("degree")
        generated_features = kwargs["output"].pop("generated_features")
        kwargs["output"] = output[: batch_mask.shape[0]]
        res = super()._compute_loss(**kwargs)
        if not self.model.training:
            return res
        degree_loss = self.__compute_degree_loss(
            batch_mask=batch_mask,
            predicated_degree=predicated_degree,
        )
        feature_loss = None
        if generated_features:
            feature_loss = self.__compute_feature_loss(
                batch_mask=batch_mask,
                generated_features=generated_features,
            )
        log_debug("degree_loss is %s", degree_loss)
        log_debug("feature_loss is %s", feature_loss)
        res["loss"] = res["loss"] + degree_loss
        if feature_loss:
            res["loss"] = res["loss"] + feature_loss
        return res


def replace_evaluator(
    executor: Executor,
    edge_index_fun: Callable | None = None,
    masked_edge_index_fun: Callable | None = None,
    masked_node_list_fun: Callable | None = None,
) -> None:
    old_model = executor.model
    if isinstance(old_model, LocalSagePlus):
        log_error("early stop replacement")
        return
    executor.model_evaluator.__class__ = FedSagePlusModelEvaluator
    executor.model_evaluator.edge_index_fun = edge_index_fun
    executor.model_evaluator.masked_edge_index_fun = masked_edge_index_fun
    executor.model_evaluator.masked_node_list_fun = masked_node_list_fun
    dc = executor.dataset_collection
    assert isinstance(dc, ClassificationDatasetCollection)
    executor.model_evaluator.set_model(
        LocalSagePlus(
            type(old_model),
            num_features=executor.dataset_util.get_original_graph(
                graph_index=0
            ).x.shape[1],
            num_classes=dc.label_number,
        )
    )
