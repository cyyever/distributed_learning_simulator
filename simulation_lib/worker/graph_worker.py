import functools
import json
import os
from typing import Any, Callable

import torch
import torch_geometric.nn
import torch_geometric.utils
from cyy_naive_lib.log import get_logger
from cyy_torch_graph import GraphDatasetUtil
from cyy_torch_toolbox.ml_type import MachineLearningPhase

from ..message import (FeatureMessage, Message, ParameterMessageBase,
                       get_message_size)
from .aggregation_worker import AggregationWorker


class GraphWorker(AggregationWorker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert not self.config.trainer_config.hook_config.use_amp
        self._share_feature = self.config.algorithm_kwargs.get("share_feature", True)
        self._other_training_node_indices: set = set()
        self.__old_edge_index: None | torch.Tensor = None
        self.__n_id: None | torch.Tensor = None
        self.__local_node_mask: None | torch.Tensor = None
        self._hook_handles: dict = {}
        self._comunicated_batch_cnt: int = 0
        self._skipped_embedding_bytes: int = 0
        self._round_skipped_bytes: dict = {}
        self._communicated_embedding_bytes: int = 0
        self._aggregated_bytes: int = 0
        self._round_communicated_bytes: dict = {}
        self._original_in_client_training_edge_cnt: int = 0
        self._in_client_training_edge_cnt: int = 0
        self._cross_client_training_edge_cnt: int = 0
        self._recorded_model_size: dict = {}
        self._send_parameter_diff = False

    def get_dataset_util(self, phase) -> GraphDatasetUtil:
        util = self.trainer.dataset_collection.get_dataset_util(phase=phase)
        assert isinstance(util, GraphDatasetUtil)
        return util

    @property
    def edge_index(self) -> torch.Tensor:
        return self.get_dataset_util(
            phase=MachineLearningPhase.Training
        ).get_edge_index(graph_index=0)

    def __get_local_edge_mask(self, edge_index: torch.Tensor) -> torch.Tensor:
        if self.__local_node_mask is None:
            if self.validation_node_mask is not None:
                self.__local_node_mask = (
                    self.training_node_mask | self.validation_node_mask
                )
            else:
                self.__local_node_mask = self.training_node_mask
        if self.__local_node_mask.device != edge_index.device:
            self.__local_node_mask = self.__local_node_mask.to(device=edge_index.device)
        return (
            self.__local_node_mask[edge_index[0]]
            & self.__local_node_mask[edge_index[1]]
        )

    def __get_local_edge_index(
        self, edge_index: torch.Tensor | None = None
    ) -> torch.Tensor:
        if edge_index is None:
            edge_index = self.edge_index
        edge_mask = self.__get_local_edge_mask(edge_index=edge_index)
        return torch_geometric.utils.coalesce(edge_index[:, edge_mask])

    def __exchange_training_node_indices(self) -> None:
        sent_data = Message(
            other_data={
                "training_node_indices": self.training_node_indices,
            },
            in_round=True,
        )
        self.send_data_to_server(sent_data)
        res = self._get_data_from_server()
        assert isinstance(res, Message)
        for worker_id, node_indices in res.other_data["training_node_indices"].items():
            if worker_id != self.worker_id:
                self._other_training_node_indices |= node_indices
        assert self._other_training_node_indices
        assert not self._other_training_node_indices.intersection(
            self.training_node_indices
        )

    def _before_training(self) -> None:
        super()._before_training()
        if self.hold_log_lock:
            if self._share_feature:
                get_logger().info("share feature")
            else:
                get_logger().info("not share feature")
        self.__exchange_training_node_indices()
        self.__clear_unrelated_edges()
        for module in self._get_message_passing_modules():
            module.register_forward_pre_hook(
                hook=self._record_embedding_size,
                with_kwargs=True,
                prepend=False,
            )
        if not self._share_feature:
            self._clear_cross_client_edges()
            return

        for module in self.trainer.model.modules():
            module.register_forward_pre_hook(
                hook=self._catch_n_id,
                with_kwargs=True,
                prepend=True,
            )
            break
        for idx, _ in enumerate(self._get_message_passing_modules()):
            if idx == 0:
                self._register_embedding_hook(
                    module_index=idx, hook=self._clear_cross_client_edge_on_the_fly
                )
            else:
                self._register_embedding_hook(
                    module_index=idx, hook=self._pass_node_feature
                )

    def _register_embedding_hook(self, module_index: int, hook: Callable) -> None:
        old_hook = self._hook_handles.pop(module_index, None)
        if old_hook is not None:
            old_hook.remove()
        self._hook_handles[module_index] = self._get_message_passing_modules()[
            module_index
        ].register_forward_pre_hook(
            hook=hook,
            with_kwargs=True,
            prepend=False,
        )

    @functools.cached_property
    def training_node_mask(self) -> torch.Tensor:
        mask = self.get_dataset_util(phase=MachineLearningPhase.Training).get_mask()
        return mask[0]

    @functools.cached_property
    def validation_node_mask(self) -> torch.Tensor | None:
        if not self.trainer.dataset_collection.has_dataset(
            phase=MachineLearningPhase.Validation
        ):
            return None
        mask = self.get_dataset_util(phase=MachineLearningPhase.Validation).get_mask()
        return mask[0]

    @functools.cached_property
    def training_node_indices(self) -> set:
        return set(
            torch_geometric.utils.mask_to_index(self.training_node_mask).tolist()
        )

    @functools.cached_property
    def training_node_boundary(self) -> set:
        assert self._other_training_node_indices
        edge_index = self.edge_index
        edge_mask = self.training_node_mask[edge_index[0]]
        edge_index = edge_index[:, edge_mask]
        worker_boundary = set()
        for a, b in edge_index.transpose(0, 1).numpy():
            if b in self._other_training_node_indices:
                worker_boundary.add(a)

        assert worker_boundary
        return worker_boundary

    @property
    def n_id(self) -> torch.Tensor:
        assert self.__n_id is not None
        return self.__n_id

    @property
    def cross_client_edge_mask(self) -> torch.Tensor:
        edge_index = self.edge_index
        return (self.training_node_mask[edge_index[0]]) & (
            torch_geometric.utils.index_to_mask(
                torch.tensor(list(self._other_training_node_indices)),
                size=edge_index.shape[1],
            )
        )[edge_index[1]]

    def __clear_unrelated_edges(self) -> None:
        assert self._other_training_node_indices
        # Keep in-device edges and cross-device edges
        edge_index = self.edge_index
        training_edge_mask = (
            self.get_dataset_util(phase=MachineLearningPhase.Training)
            .get_edge_masks()[0]
            .clone()
        )
        self._original_in_client_training_edge_cnt = int(
            training_edge_mask.sum().item()
        )
        edge_drop_rate: float | None = self.config.algorithm_kwargs.get(
            "edge_drop_rate", None
        )
        if edge_drop_rate is not None and edge_drop_rate != 0:
            if self.hold_log_lock:
                get_logger().info("drop in client edge with rate: %s", edge_drop_rate)
            dropout_mask = torch.bernoulli(
                torch.full(training_edge_mask.size(), 1 - edge_drop_rate)
            ).to(dtype=torch.bool)
            training_edge_mask &= dropout_mask

        if training_edge_mask.sum().item() != 0:
            get_logger().info(
                "cross_client_edge/in_client_edge %s",
                self.cross_client_edge_mask.sum().item()
                / training_edge_mask.sum().item(),
            )
        self._cross_client_training_edge_cnt = int(
            self.cross_client_edge_mask.sum().item()
        )
        self._in_client_training_edge_cnt = int(training_edge_mask.sum().item())
        edge_mask = training_edge_mask | self.cross_client_edge_mask
        if self.validation_node_mask is not None:
            validation_edge_mask = (
                self.validation_node_mask[edge_index[0]]
                & self.validation_node_mask[edge_index[1]]
            )
            edge_mask = edge_mask | validation_edge_mask
        edge_index = torch_geometric.utils.coalesce(edge_index[:, edge_mask])
        self.trainer.dataset_collection.transform_dataset(
            self.trainer.phase,
            lambda _, dataset_util, __: dataset_util.get_edge_subset(
                graph_index=0, edge_index=edge_index
            ),
        )

    def _clear_cross_client_edges(self) -> None:
        edge_index = self.__get_local_edge_index()
        self.trainer.dataset_collection.transform_dataset(
            self.trainer.phase,
            lambda _, dataset_util, __: dataset_util.get_edge_subset(
                graph_index=0, edge_index=edge_index
            ),
        )

    def _clear_cross_client_edge_on_the_fly(
        self, module: torch.nn.Module, args: Any, kwargs: Any
    ) -> tuple | None:
        if not module.training:
            return None
        self.__old_edge_index = args[1]
        assert self.__old_edge_index is not None
        edge_index = self.n_id[self.__old_edge_index]
        edge_mask = self.__get_local_edge_mask(edge_index=edge_index)
        args = list(args)
        args[1] = self.__old_edge_index[:, edge_mask]
        x = args[0]
        cnt = len(self.training_node_boundary.intersection(set(self.n_id.tolist())))
        assert len(x.shape) == 2
        self._skipped_embedding_bytes += x.shape[1] * x.element_size() * cnt

        return tuple(args), kwargs

    def training_boundary_feature(self, x) -> tuple[torch.Tensor, list] | None:
        assert len(self.training_node_boundary) <= len(self.training_node_indices)

        assert x.shape[0] == self.n_id.shape[0]
        indices: list = []
        node_indices = []
        for idx, node_index in enumerate(self.n_id.tolist()):
            if node_index in self.training_node_boundary:
                indices.append(idx)
                node_indices.append(node_index)
        if not indices:
            return None
        return (
            torch.index_select(x, 0, torch.tensor(indices, device=x.device))
            .detach()
            .cpu(),
            node_indices,
        )

    def _get_cross_deivce_embedding(
        self, embedding_indices, embedding, x
    ) -> torch.Tensor:
        if not embedding_indices:
            return x
        new_x = torch.zeros_like(x)
        mask1 = torch.zeros_like(x, dtype=torch.bool)
        mask2 = torch.zeros_like(x, dtype=torch.bool)
        embedding_mask = torch.zeros((embedding.shape[0]), dtype=torch.bool)

        embedding_indices = {b: a for a, b in enumerate(embedding_indices)}

        for idx, node_idx in enumerate(self.n_id.tolist()):
            if node_idx not in self.training_node_indices:
                if node_idx in embedding_indices:
                    assert node_idx not in self.training_node_indices
                    mask1[idx] = True
                    embedding_mask[embedding_indices[node_idx]] = True
                else:
                    mask2[idx] = True
        new_x[mask1[:, 0]] = embedding[embedding_mask].to(
            new_x.device, non_blocking=True
        )
        mask = mask1 | mask2
        new_x = torch.where(mask, new_x, x)
        return new_x

    def _catch_n_id(self, module, args, kwargs) -> tuple | None:
        self.__n_id = kwargs["n_id"]
        return args, kwargs

    def _record_embedding_size(
        self,
        module,
        args,
        kwargs,
    ) -> tuple | None:
        if "model_bytes" not in self._recorded_model_size:
            parameter = self.trainer.model_util.get_parameter_list()
            self._recorded_model_size["model_bytes"] = (
                parameter.element_size() * parameter.numel()
            )
        if "embedding_bytes" not in self._recorded_model_size:
            self._recorded_model_size["embedding_bytes"] = []
        if len(self._recorded_model_size["embedding_bytes"]) < len(
            self._get_message_passing_modules()
        ):
            x = args[0]
            self._recorded_model_size["embedding_bytes"].append(
                x[0].numel() * x[0].element_size()
            )
        return None

    def _pass_node_feature(self, module, args, kwargs) -> tuple | None:
        if not module.training:
            return None

        self.trainer.wait_stream()

        x = args[0]

        n_id_set = set(self.n_id.tolist())
        boundary = self._other_training_node_indices.intersection(n_id_set)
        res = self.training_boundary_feature(x)
        if res is not None:
            feature = res[0]
            sent_data = FeatureMessage(
                feature=feature,
                other_data={
                    "boundary": boundary,
                    "node_indices": res[1],
                },
                in_round=True,
            )
            self._communicated_embedding_bytes += (
                feature.numel() * feature.element_size()
            )
        else:
            assert not boundary
            sent_data = FeatureMessage(
                feature=None,
                other_data={
                    "boundary": boundary,
                },
                in_round=True,
            )
        self._comunicated_batch_cnt += 1
        self.send_data_to_server(sent_data)
        res = self._get_data_from_server()
        assert isinstance(res, FeatureMessage)

        new_x = self._get_cross_deivce_embedding(
            res.other_data["node_indices"], res.feature, x
        )
        self.__n_id = None
        return (new_x, self.__old_edge_index, *args[2:]), kwargs

    def _get_message_passing_modules(self) -> list:
        return [
            module
            for module in self.trainer.model.modules()
            if isinstance(module, torch_geometric.nn.MessagePassing)
        ]

    def _get_sent_data(self) -> ParameterMessageBase:
        sent_data = super()._get_sent_data()
        self._aggregated_bytes += get_message_size(sent_data)
        self._round_communicated_bytes[self._round_index] = (
            self._aggregated_bytes + self._communicated_embedding_bytes
        )
        self._round_skipped_bytes[self._round_index] = self._skipped_embedding_bytes
        return sent_data

    def _after_training(self) -> None:
        super()._after_training()
        with open(
            os.path.join(self.save_dir, "graph_worker_stat.json"), "wt", encoding="utf8"
        ) as f:
            stat = {
                "original_in_client_training_edge_cnt": self._original_in_client_training_edge_cnt,
                "in_client_training_edge_cnt": self._in_client_training_edge_cnt,
                "cross_client_training_edge_cnt": self._cross_client_training_edge_cnt,
                "training_node_cnt": self.training_node_mask.sum().item(),
                "skipped_embedding_bytes": self._round_skipped_bytes,
                "communicated_bytes": self._round_communicated_bytes,
            }
            if self.validation_node_mask is not None:
                stat["validation_node_cnt"] = self.validation_node_mask.sum().item()
            json.dump(stat | self._recorded_model_size, f)
