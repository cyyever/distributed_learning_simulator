import random

import torch
import torch_geometric.utils
from cyy_naive_lib.log import log_debug
from cyy_torch_toolbox import MachineLearningPhase
from distributed_learning_simulation.graph_worker import GraphWorker

from .evaluator import replace_evaluator


class FedSagePlusWorker(GraphWorker):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__before_training_flag = True
        self._share_feature = False
        self._remove_cross_edge = True
        self.__hiddien_portion = self.config.algorithm_kwargs.get(
            "hidden_portion", 0.25
        )
        self.__original_training_nodes: set = set()
        self.__original_edge_index: None | torch.Tensor = None
        self.__masked_edge_index: None | torch.Tensor = None
        self.__hidden_node_indices: list = []
        self.trainer.hook_config.use_slow_performance_metrics = False

    def __get_edge_index(self) -> torch.Tensor:
        assert self.__original_edge_index is not None
        return self.__original_edge_index

    def __get_masked_edge_index(self) -> torch.Tensor:
        assert self.__masked_edge_index is not None
        return self.__masked_edge_index

    def __get_hidden_node_indices(self) -> list:
        return self.__hidden_node_indices

    def __hide_training_nodes(self) -> None:
        log_debug("hid training_node_indices")
        if not self.__original_training_nodes:
            self.__original_training_nodes = set(self.training_node_indices)
        if self.__original_edge_index is None:
            self.__original_edge_index = self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_edge_index(graph_index=0)

        indices = list(self.__original_training_nodes)
        hidden_indices = random.sample(
            indices, k=int(len(indices) * self.__hiddien_portion)
        )
        assert hidden_indices
        self.__hidden_node_indices = hidden_indices
        self.trainer.mutable_dataset_collection.set_subset(
            phase=MachineLearningPhase.Training,
            indices=self.__original_training_nodes - set(hidden_indices),
        )
        mask = self.get_dataset_util().get_mask()[0].tolist()
        assert not mask[list(hidden_indices)[0]]
        assert not mask[list(hidden_indices)[-1]]

        self.__masked_edge_index = torch_geometric.utils.coalesce(
            self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_masked_edge_index(
                graph_index=0, edge_index=self.__original_edge_index
            )
        )
        self.trainer.mutable_dataset_collection.transform_dataset(
            self.trainer.phase,
            lambda _, dataset_util, __: dataset_util.get_edge_subset(
                graph_index=0, edge_index=self.__masked_edge_index
            ),
        )

    def _before_training(self) -> None:
        replace_evaluator(
            self.trainer,
            edge_index_fun=self.__get_edge_index,
            masked_edge_index_fun=self.__get_masked_edge_index,
            masked_node_list_fun=self.__get_hidden_node_indices,
        )
        super()._before_training()
        log_debug("end _before_training")
        self.__before_training_flag = False
        self.__hide_training_nodes()

    def _load_result_from_server(self, *args, **kwargs) -> None:
        if not self.__before_training_flag:
            self.__hide_training_nodes()
        super()._load_result_from_server(*args, **kwargs)
