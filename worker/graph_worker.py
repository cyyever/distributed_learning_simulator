import functools
from typing import Any

import torch
import torch_geometric.nn
import torch_geometric.utils
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.tensor import tensor_to

from worker.fed_avg_worker import FedAVGWorker


class GraphWorker(FedAVGWorker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert not self.config.trainer_config.hook_config.use_amp
        self._share_feature = self.config.algorithm_kwargs.get("share_feature", True)
        self._other_training_node_indices: set = set()
        self.__old_edge_index = None
        self.__n_id: None | torch.Tensor = None
        self._hook_handles: dict = {}

    def __get_edge_index(self) -> torch.Tensor:
        return self.trainer.dataset_collection.get_dataset_util(
            phase=MachineLearningPhase.Training
        ).get_edge_index(graph_index=0)

    def __get_local_edge_index(
        self, edge_index: None | torch.Tensor = None
    ) -> torch.Tensor:
        if edge_index is None:
            edge_index = self.__get_edge_index()
        edge_mask = self.__get_local_edge_mask(edge_index=edge_index)
        assert edge_index is not None
        return torch_geometric.utils.coalesce(edge_index[:, edge_mask])

    def __get_local_edge_mask(self, edge_index) -> torch.Tensor:
        local_node_mask = self.training_node_mask | self.validation_node_mask
        local_node_mask = tensor_to(local_node_mask, device=edge_index.device)
        return local_node_mask[edge_index[0]] & local_node_mask[edge_index[1]]

    def __exchange_training_node_indices(self) -> None:
        sent_data = {
            "training_node_indices": self.training_node_indices,
        }
        self.send_data_to_server(sent_data)
        res = self._get_data_from_server()
        assert res is not None
        for worker_id, node_indices in res["training_node_indices"].items():
            if worker_id != self.worker_id:
                self._other_training_node_indices |= node_indices
        assert self._other_training_node_indices
        assert not self._other_training_node_indices.intersection(
            self.training_node_indices
        )

    def _before_training(self) -> None:
        super()._before_training()
        if self._share_feature:
            get_logger().warning("share feature")
        else:
            get_logger().warning("not share feature")
        self.__exchange_training_node_indices()
        self.__clear_unrelated_edges()

        self._determin_batch_size()
        if not self._share_feature:
            self._clear_cross_device_edges()
            return

        # We need to get training and neighbor nodes
        input_nodes = torch.tensor(
            list(
                set(self.__get_edge_index().view(-1).tolist())
                - set(
                    torch_geometric.utils.mask_to_index(
                        self.validation_node_mask
                    ).tolist()
                )
            ),
            dtype=torch.long,
        )
        self.trainer.hyper_parameter.extra_parameters["pyg_input_nodes"] = {
            MachineLearningPhase.Training: input_nodes
        }
        for module in self.trainer.model.modules():
            module.register_forward_pre_hook(
                hook=self._catch_n_id,
                with_kwargs=True,
                prepend=True,
            )
            break
        for idx, module in enumerate(self._get_message_passing_modules()):
            hook = self._pass_node_feature
            if idx == 0:
                hook = self._clear_cross_device_edge_on_the_fly
            self._register_embedding_hook(idx=idx, hook=hook)

    def _register_embedding_hook(self, idx, hook) -> None:
        old_hook = self._hook_handles.pop(idx, None)
        if old_hook is not None:
            old_hook.remove()
        self._hook_handles[idx] = self._get_message_passing_modules()[
            idx
        ].register_forward_pre_hook(
            hook=hook,
            with_kwargs=True,
            prepend=False,
        )

    @functools.cached_property
    def training_node_mask(self) -> torch.Tensor:
        mask = self.trainer.dataset_collection.get_dataset_util(
            phase=MachineLearningPhase.Training
        ).get_mask()
        assert mask is not None
        return mask[0]

    @functools.cached_property
    def validation_node_mask(self) -> torch.Tensor:
        mask = self.trainer.dataset_collection.get_dataset_util(
            phase=MachineLearningPhase.Validation
        ).get_mask()
        assert mask is not None
        return mask[0]

    @functools.cached_property
    def training_node_indices(self) -> set:
        return set(
            torch_geometric.utils.mask_to_index(self.training_node_mask).tolist()
        )

    @functools.cached_property
    def training_node_boundary(self) -> set:
        assert self._other_training_node_indices
        edge_index = self.__get_edge_index()
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

    def __clear_unrelated_edges(self) -> None:
        assert self._other_training_node_indices
        # Keep in-device edges and cross-device edges
        edge_index = self.__get_edge_index()
        in_client_edge_mask = (
            self.training_node_mask[edge_index[0]]
            & self.training_node_mask[edge_index[1]]
        )
        edge_drop_rate: float | None = self.config.algorithm_kwargs.get(
            "edge_drop_rate", None
        )
        if edge_drop_rate is not None:
            dropout_mask = torch.bernoulli(
                torch.full(in_client_edge_mask.size(), 1 - edge_drop_rate)
            ).to(dtype=torch.bool)
            in_client_edge_mask &= dropout_mask

        cross_device_edge_mask = (self.training_node_mask[edge_index[0]]) & (
            torch_geometric.utils.index_to_mask(
                torch.tensor(list(self._other_training_node_indices)),
                size=self.training_node_mask.shape[0],
            )
        )[edge_index[1]]

        validation_edge_mask = (
            self.validation_node_mask[edge_index[0]]
            & self.validation_node_mask[edge_index[1]]
        )
        get_logger().warning(
            "cross_device_edge/in_client_edge %s",
            cross_device_edge_mask.sum().item() / in_client_edge_mask.sum().item(),
        )
        joint_mask = in_client_edge_mask | cross_device_edge_mask | validation_edge_mask
        edge_index = torch_geometric.utils.coalesce(edge_index[:, joint_mask])
        self.trainer.transform_dataset(
            lambda _, dataset_util, __: dataset_util.get_edge_subset(
                graph_index=0, edge_index=edge_index
            )
        )

    def _clear_cross_device_edges(self) -> None:
        edge_index = self.__get_local_edge_index()
        self.trainer.transform_dataset(
            lambda _, dataset_util, __: dataset_util.get_edge_subset(
                graph_index=0, edge_index=edge_index
            )
        )

    def _clear_cross_device_edge_on_the_fly(
        self, module: torch.nn.Module, args: Any, kwargs: Any
    ) -> tuple | None:
        if not module.training:
            return None
        self.__old_edge_index = args[1]
        edge_index = self.n_id[self.__old_edge_index]
        edge_mask = self.__get_local_edge_mask(edge_index=edge_index)
        args = list(args)
        args[1] = args[1][:, edge_mask]
        return tuple(args), kwargs

    def training_boundary_feature(self, x) -> tuple:
        assert len(self.training_node_boundary) <= len(self.training_node_indices)

        assert x.shape[0] == self.n_id.shape[0]
        indices: list = []
        node_indices = []
        for idx, node_index in enumerate(self.n_id.tolist()):
            if node_index in self.training_node_boundary:
                indices.append(idx)
                node_indices.append(node_index)
        assert indices
        return (
            torch.index_select(x, 0, torch.tensor(indices, device=x.device))
            .detach()
            .cpu(),
            node_indices,
        )

    def _get_cross_deivce_embedding(
        self, embedding_indices, embedding, x
    ) -> torch.Tensor:
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
        # get_logger().error(
        #     "embedding_indices %s mask1 %s mask2 %s",
        #     len(embedding_indices),
        #     mask1.sum(),
        #     mask2.sum(),
        # )
        new_x[mask1[:, 0]] = embedding[embedding_mask].to(
            new_x.device, non_blocking=True
        )
        mask = mask1 | mask2
        new_x = torch.where(mask, new_x, x)
        return new_x

    def _catch_n_id(self, module, args, kwargs) -> tuple | None:
        self.__n_id = kwargs.pop("n_id", None)
        return args, kwargs

    def _pass_node_feature(self, module, args, kwargs) -> tuple | None:
        if not module.training:
            return None

        self.trainer.wait_stream()

        x = args[0]

        sent_data = {
            "node_embedding": self.training_boundary_feature(x),
            "boundary": set(self.n_id.tolist()) - self.training_node_indices,
        }
        self.send_data_to_server(sent_data)
        res = self._get_data_from_server()
        assert res is not None

        new_x = self._get_cross_deivce_embedding(
            res["node_indices"], res["node_embedding"], x
        )
        self.__n_id = None
        return (new_x, self.__old_edge_index, *args[2:]), kwargs

    def _determin_batch_size(self):
        self.trainer.hyper_parameter.extra_parameters[
            "batch_number"
        ] = self.config.algorithm_kwargs["batch_number"]

    def _get_message_passing_modules(self) -> list:
        return [
            module
            for module in self.trainer.model.modules()
            if isinstance(module, torch_geometric.nn.MessagePassing)
        ]
