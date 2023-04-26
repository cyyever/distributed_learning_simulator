import functools

import torch
import torch_geometric.nn
import torch_geometric.utils
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import MachineLearningPhase

from worker.fed_avg_worker import FedAVGWorker


class GraphWorker(FedAVGWorker):
    def __init__(self, **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self._share_init_node_feature = self.config.algorithm_kwargs.get(
            "share_init_node_feature", False
        )
        self._share_node_feature = self.config.algorithm_kwargs.get(
            "share_node_feature", True
        )
        self._masked_init_features = {}

    def _before_training(self) -> None:
        first_message_layer: bool = True
        for module in self.trainer.model.modules():
            if not isinstance(module, torch_geometric.nn.MessagePassing):
                continue
            if first_message_layer:
                if not self._share_init_node_feature:
                    module.register_forward_pre_hook(
                        hook=self._clear_init_node_feature,
                        with_kwargs=True,
                        prepend=False,
                    )
            else:
                if self._share_node_feature:
                    module.register_forward_pre_hook(
                        hook=self._pass_node_feature, with_kwargs=True, prepend=False
                    )
            first_message_layer = False
        self._masked_init_features = {}
        if self._share_init_node_feature:
            get_logger().warning("share init node feature")
        else:
            get_logger().warning("not share init node feature")
        if self._share_node_feature:
            get_logger().warning("share node feature")
        else:
            get_logger().warning("not share node feature")
        super()._before_training()

    @functools.cached_property
    def training_mask(self) -> torch.Tensor:
        return self.trainer.dataset[0]["subset_mask"]

    @functools.cached_property
    def validation_mask(self) -> torch.Tensor:
        return self.trainer.dataset_collection.get_dataset(
            phase=MachineLearningPhase.Validation
        )[0]["subset_mask"]

    def __in_training(self, module_args) -> bool:
        return module_args[0].requires_grad

    @functools.cached_property
    def training_node_indices(self) -> set:
        return set(torch_geometric.utils.mask_to_index(self.training_mask).tolist())

    @functools.cached_property
    def validation_node_indices(self) -> set:
        return set(torch_geometric.utils.mask_to_index(self.validation_mask).tolist())

    @functools.cached_property
    def training_node_boundary(self) -> dict:
        worker_boundary = self.trainer.dataset_util.get_boundary(
            self.training_node_indices | self.validation_node_indices
        )
        return {
            k: v for k, v in worker_boundary.items() if k in self.training_node_indices
        }

    def _clear_init_node_feature(self, module, args, kwargs) -> tuple | None:
        if not self.__in_training(args):
            return None
        phase = MachineLearningPhase.Training

        if phase not in self._masked_init_features:
            self._masked_init_features[phase] = torch.zeros_like(args[0])
            mask = torch.zeros(
                (self._masked_init_features[phase].shape[0],), dtype=torch.bool
            )
            assert self.trainer.model_evaluator.node_and_neighbour_index_map[phase]
            index_map = self.trainer.model_evaluator.node_and_neighbour_index_map[phase]
            if phase == MachineLearningPhase.Training:
                for node_index in self.training_node_indices:
                    mask[index_map[node_index]] = True
            else:
                for node_index in self.validation_node_indices:
                    mask[index_map[node_index]] = True
            self._masked_init_features[phase][mask] = args[0][mask]

        return (self._masked_init_features[phase], *args[1:]), kwargs

    def _pass_node_feature(self, module, args, kwargs) -> tuple | None:
        if not self.__in_training(args):
            return None

        self.trainer.wait_stream()
        assert len(self.training_node_boundary) <= len(self.training_node_indices)
        phase = MachineLearningPhase.Training

        index_map = self.trainer.model_evaluator.node_and_neighbour_index_map[phase]
        x = args[0]
        assert x.shape[0] == len(index_map)
        sent_data = {
            "node_embedding": torch.stack(
                [
                    x[index_map[node_index]]
                    for node_index in sorted(self.training_node_boundary.keys())
                ]
            )
            .detach()
            .cpu(),
            "boundary": self.training_node_boundary,
            "in_round_data": True,
        }
        self.send_data_to_server(sent_data)
        res = self._get_result_from_server()
        new_x = torch.zeros_like(x)
        mask = torch.zeros_like(x, dtype=torch.bool)

        for idx, node_index in enumerate(res["boundary"]):
            new_embedding = res["node_embedding"][idx].to(
                new_x.device, non_blocking=True
            )
            idx = index_map[node_index]
            mask[idx] = True
            new_x[idx] = new_embedding
        new_x = torch.where(mask, x, new_x)
        return (new_x, *args[1:]), kwargs
