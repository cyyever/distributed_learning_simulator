import functools
import math
from typing import Any

import torch
import torch_geometric.nn
import torch_geometric.utils
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.dataset_util import GraphDatasetUtil
from cyy_torch_toolbox.ml_type import MachineLearningPhase

from worker.fed_avg_worker import FedAVGWorker


class GraphWorker(FedAVGWorker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert not self.config.trainer_config.hook_config.use_amp
        self._share_feature = self.config.algorithm_kwargs.get("share_feature", True)
        self._other_training_node_indices: set = set()
        self._other_training_node_mask: dict = {}

    def _before_training(self) -> None:
        super()._before_training()
        batch_number = self.config.algorithm_kwargs["batch_number"]
        dataset_size = len(self.training_node_indices)
        self.trainer.hyper_parameter.batch_size = math.ceil(dataset_size / batch_number)
        assert self.trainer.hyper_parameter.batch_size > 0
        assert self.trainer.hyper_parameter.get_iterations_per_epoch(
            dataset_size=dataset_size
        )
        if not self._share_feature:
            self._clear_cross_device_edges()
        else:
            first_message_layer: bool = True
            for module in self.trainer.model.modules():
                if not isinstance(module, torch_geometric.nn.MessagePassing):
                    continue
                if first_message_layer:
                    module.register_forward_pre_hook(
                        hook=self._clear_init_node_feature,
                        with_kwargs=True,
                        prepend=False,
                    )
                else:
                    module.register_forward_pre_hook(
                        hook=self._pass_node_feature,
                        with_kwargs=True,
                        prepend=False,
                    )
                first_message_layer = False
        if self._share_feature:
            get_logger().warning("share feature")
        else:
            get_logger().warning("not share feature")
        sent_data = {
            "training_node_indices": self.training_node_indices,
            "in_round_data": True,
        }
        self.send_data_to_server(sent_data)
        res = self._get_result_from_server()
        for worker_id, node_indices in res["training_node_indices"].items():
            if worker_id != self.worker_id:
                self._other_training_node_indices |= node_indices

    @functools.cached_property
    def training_node_indices(self) -> set:
        return set(
            torch_geometric.utils.mask_to_index(
                self.trainer.dataset[0]["mask"]
            ).tolist()
        )

    @functools.cached_property
    def validation_node_indices(self) -> set:
        return set(
            torch_geometric.utils.mask_to_index(
                self.trainer.dataset_collection.get_dataset(
                    phase=MachineLearningPhase.Validation
                )[0]["mask"]
            ).tolist()
        )

    @functools.cached_property
    def training_node_boundary(self) -> dict:
        worker_boundary: dict = self.trainer.dataset_util.get_boundary(
            self.training_node_indices
        )
        assert self._other_training_node_indices
        res = {
            k: v.intersection(self._other_training_node_indices)
            for k, v in worker_boundary.items()
            if k in self.training_node_indices
        }
        res = {k: v for k, v in res.items() if v}
        self._other_training_node_indices.clear()
        return res

    def _clear_cross_device_edges(self) -> None:
        subset_node_indices: set = (
            self.training_node_indices | self.validation_node_indices
        )
        edge_indices = []
        cnt = 0
        edge_index = self.trainer.dataset_util.get_edge_index(0)
        for idx, edge in enumerate(GraphDatasetUtil.foreach_edge(edge_index)):
            if (
                edge[0] not in subset_node_indices
                and edge[1] not in subset_node_indices
            ):
                cnt += 1
                continue
            edge_indices.append(idx)
        assert cnt > 0
        print(cnt)
        assert edge_indices
        self.trainer.transform_dataset(
            lambda _, dataset_util, __: dataset_util.get_edge_subset(edge_indices)
        )

    def _clear_init_node_feature(
        self, module: torch.nn.Module, args: Any, kwargs: Any
    ) -> tuple | None:
        if not module.training:
            return None
        phase = MachineLearningPhase.Training
        x = args[0]
        masked_init_features = torch.zeros_like(x)
        mask = torch.zeros((x.shape[0],), dtype=torch.bool)
        index_map = self.trainer.model_evaluator.batch_neighbour_index_map[phase]
        flag = False
        for node_index in self.training_node_indices:
            if node_index in index_map:
                mask[index_map[node_index]] = True
                flag = True
        assert flag
        masked_init_features[mask] = x[mask]

        return (masked_init_features, *args[1:]), kwargs

    def training_node_boundary_mask(self) -> torch.Tensor:
        assert len(self.training_node_boundary) <= len(self.training_node_indices)

        index_map = self.trainer.model_evaluator.batch_neighbour_index_map[
            MachineLearningPhase.Training
        ]
        shape = len(index_map)
        mask = torch.zeros((shape,), dtype=torch.bool)
        for node_index in self.training_node_boundary:
            if node_index in index_map:
                mask[index_map[node_index]] = True
        print(mask.sum().item())
        return mask

    def __get_other_training_node_mask(self, node_indices, feature) -> torch.Tensor:
        mask = torch.zeros_like(feature, dtype=torch.bool)
        index_map = self.trainer.model_evaluator.batch_neighbour_index_map[
            MachineLearningPhase.Training
        ]
        for node_index in node_indices:
            mask[index_map[node_index]] = True
        return mask

    def _pass_node_feature(self, module, args, kwargs) -> tuple | None:
        if not module.training:
            return None

        self.trainer.wait_stream()

        x = args[0]

        sent_data = {
            "node_embedding": x[self.training_node_boundary_mask()].cpu().detach(),
            "boundary": self.training_node_boundary,
            "in_round_data": True,
        }
        self.send_data_to_server(sent_data)
        res = self._get_result_from_server()

        mask = self.__get_other_training_node_mask(res["node_indices"], x)
        new_x = torch.zeros_like(x)
        new_x[mask[:, 0]] = res["node_embedding"].to(new_x.device, non_blocking=True)
        new_x = torch.where(mask, new_x, x)
        return (new_x, *args[1:]), kwargs
