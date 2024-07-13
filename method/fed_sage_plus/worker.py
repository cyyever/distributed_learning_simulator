import random

import torch
import torch_geometric.utils
from cyy_torch_toolbox import MachineLearningPhase
from distributed_learning_simulation import GraphWorker

from .evaluator import replace_evaluator


class FedSagePlusWorker(GraphWorker):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._share_feature = False
        self._remove_cross_edge = True
        self.__hiddien_portion = self.config.algorithm_kwargs.get(
            "hidden_portion", 0.25
        )
        self.__original_training_nodes: set = set()
        self.__masked_edge_index: None | torch.Tensor = None

    def __get_masked_edge_index(self) -> torch.Tensor:
        assert self.__masked_edge_index is not None
        return self.__masked_edge_index

    def __hide_training_nodes(self) -> None:
        if not self.__original_training_nodes:
            self.__original_training_nodes = set(self.training_node_indices)
        original_edge_index = (
            self.get_dataset_util(phase=MachineLearningPhase.Training)
            .get_original_graph(0)
            .edge_index
        )

        indices = list(self.__original_training_nodes)
        hidden_indices = random.sample(
            indices, k=int(len(indices) * self.__hiddien_portion)
        )
        self.trainer.mutable_dataset_collection.set_subset(
            phase=MachineLearningPhase.Training,
            indices=self.__original_training_nodes - set(hidden_indices),
        )

        self.__masked_edge_index = torch_geometric.utils.coalesce(
            self.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).get_masked_edge_index(graph_index=0, edge_index=original_edge_index)
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
            masked_edge_index_fun=self.__get_masked_edge_index,
        )
        super()._before_training()

    def _load_result_from_server(self, *args, **kwargs) -> None:
        self.__hide_training_nodes()
        super()._load_result_from_server(*args, **kwargs)
