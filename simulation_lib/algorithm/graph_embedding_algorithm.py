import torch

from ..message import FeatureMessage, Message, MultipleWorkerMessage
from .aggregation_algorithm import AggregationAlgorithm


class GraphNodeEmbeddingPassingAlgorithm(AggregationAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.__node_embeddings: list[torch.Tensor] = []
        self.__node_embedding_indices: dict = {}
        self.__boundaris: dict = {}

    def process_worker_data(self, worker_id: int, worker_data: Message | None) -> bool:
        if isinstance(worker_data, FeatureMessage):
            node_embedding = worker_data.feature
            if node_embedding is not None:
                node_indices = worker_data.other_data.pop("node_indices")
                for tensor_idx, node_idx in enumerate(node_indices):
                    assert node_idx not in self.__node_embedding_indices
                    self.__node_embedding_indices[node_idx] = (
                        len(self.__node_embeddings),
                        tensor_idx,
                    )
                self.__node_embeddings.append(node_embedding)
            self.__boundaris[worker_id] = worker_data.other_data.pop("boundary")
            assert not worker_data.other_data
            return True
        return False

    def __get_node_embedding(self, node_idx) -> torch.Tensor:
        list_idx, tensor_idx = self.__node_embedding_indices[node_idx]
        return self.__node_embeddings[list_idx][tensor_idx]

    def aggregate_worker_data(self) -> Message:
        assert self.__node_embeddings or self.__boundaris
        worker_data: dict[int, FeatureMessage] = {}
        node_embedding_index_set = set(self.__node_embedding_indices.keys())
        for worker_id, boundary in self.__boundaris.items():
            node_indices = boundary.intersection(node_embedding_index_set)
            node_indices = tuple(sorted(node_indices))
            node_embedding = None
            if node_indices:
                node_embedding = torch.stack(
                    [
                        self.__get_node_embedding(node_idx).cpu()
                        for node_idx in node_indices
                    ]
                )
            worker_data[worker_id] = FeatureMessage(
                feature=node_embedding,
                other_data={"node_indices": node_indices},
                in_round=True,
            )
        return MultipleWorkerMessage(in_round=True, worker_data=worker_data)

    def clear_worker_data(self) -> None:
        self.__node_embeddings = []
        self.__node_embedding_indices = {}
        self.__boundaris = {}
