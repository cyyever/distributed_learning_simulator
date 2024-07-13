from functools import cached_property
from typing import Any, Callable

import torch
import torch.nn
import torch.nn.functional as F
from cyy_naive_lib.log import log_error
from cyy_torch_graph import GraphModelEvaluator
from cyy_torch_toolbox import (ClassificationDatasetCollection, Executor,
                               MachineLearningPhase)
from torch_geometric.utils import degree as degree_fun

from .model import LocalSagePlus


class FedSagePlusModelEvaluator(GraphModelEvaluator):
    masked_edge_index_fun: Callable | None = None
    x: torch.Tensor | None = None

    def __compute_degree_loss(
        self, batch_mask: torch.Tensor, predicated_degree: torch.Tensor, n_id_list: list
    ) -> torch.Tensor:
        assert self.masked_edge_index_fun is not None
        masked_edge_index = self.masked_edge_index_fun()
        original_degree = degree_fun(index=self.old_edge_index[0])
        masked_degree = degree_fun(
            index=masked_edge_index[0], num_nodes=original_degree.shape[0]
        )
        degree = predicated_degree.view(-1)
        degree_diff = torch.zeros_like(degree)
        for idx, n_id in enumerate(n_id_list):
            degree_diff[idx] = original_degree[n_id] - masked_degree[n_id]
        return F.smooth_l1_loss(degree_diff[batch_mask], degree[batch_mask])

    @cached_property
    def old_edge_index(self) -> torch.Tensor:
        return (
            self.get_dataset_util(phase=MachineLearningPhase.Training)
            .get_original_graph(0)
            .edge_index
        )

    def __compute_feature_loss(
        self,
        batch_mask: torch.Tensor,
        n_id_list: list,
        generated_features: dict,
    ) -> torch.Tensor:
        assert self.masked_edge_index_fun is not None
        masked_edge_index = self.masked_edge_index_fun()
        batch_mask_list = batch_mask.tolist()
        l2_sum: None | torch.Tensor = None
        if self.x is None:
            self.x = (
                self.get_dataset_util(phase=MachineLearningPhase.Training)
                .get_original_graph(0)
                .x.to(device=batch_mask.device)
            )
        assert self.x is not None
        for idx in generated_features:
            if not batch_mask_list[idx]:
                continue
            hidden_neighbor_set = set(
                self.old_edge_index[
                    1, self.old_edge_index[0] == n_id_list[idx]
                ].tolist()
            ) - set(
                masked_edge_index[1, masked_edge_index[0] == n_id_list[idx]].tolist()
            )
            if not hidden_neighbor_set:
                continue
            for filled_feature in generated_features[idx].values():
                min_diff = None
                for hidden_neighbor in hidden_neighbor_set:
                    diff = torch.nn.MSELoss(reduction="sum")(
                        self.x[hidden_neighbor], filled_feature
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
        assert l2_sum is not None
        return l2_sum

    def _compute_loss(self, **kwargs: Any) -> dict:
        batch_mask = kwargs["batch_mask"]
        output = kwargs["output"]["output"]
        predicated_degree = kwargs["output"].pop("degree")
        generated_features = kwargs["output"].pop("generated_features")
        n_id_list = kwargs["n_id"].tolist()
        kwargs["output"] = output[: batch_mask.shape[0]]
        res = super()._compute_loss(**kwargs)
        degree_loss = self.__compute_degree_loss(
            batch_mask=batch_mask,
            predicated_degree=predicated_degree,
            n_id_list=n_id_list,
        )
        feature_loss = None
        if generated_features:
            feature_loss = self.__compute_feature_loss(
                batch_mask=batch_mask,
                n_id_list=n_id_list,
                generated_features=generated_features,
            )
        log_error("degree_loss is %s", degree_loss)
        log_error("feature_loss is %s", feature_loss)
        res["loss"] = res["loss"] + degree_loss
        if feature_loss:
            res["loss"] = res["loss"] + feature_loss
        return res


def replace_evaluator(
    executor: Executor,
    masked_edge_index_fun: Callable | None = None,
) -> None:
    executor.model_evaluator.__class__ = FedSagePlusModelEvaluator
    executor.model_evaluator.masked_edge_index_fun = masked_edge_index_fun
    old_model = executor.model
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
