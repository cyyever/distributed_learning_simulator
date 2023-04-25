import torch
from cyy_naive_lib.storage import DataStorage

from .fed_avg_algorithm import FedAVGAlgorithm


class GraphNodeEmbeddingPassingAlgorithm(FedAVGAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__node_embeddings: list = []
        self.__node_embedding_indices: dict = {}
        self.__boundaris: dict = {}

    def process_worker_data(
        self,
        worker_id: int,
        worker_data: dict[str, DataStorage],
        old_parameter_dict: dict | None,
        save_dir: str,
    ) -> None:
        super().process_worker_data(
            worker_id=worker_id,
            worker_data=worker_data,
            old_parameter_dict=old_parameter_dict,
            save_dir=save_dir,
        )
        worker_data = self._all_worker_data[worker_id].data
        if worker_data is None:
            return
        node_embedding = worker_data.pop("node_embedding", None)
        if node_embedding is not None:
            boundary = worker_data.pop("boundary")
            for tensor_idx, node_idx in enumerate(sorted(boundary.keys())):
                assert node_idx not in self.__node_embedding_indices
                self.__node_embedding_indices[node_idx] = (
                    len(self.__node_embeddings),
                    tensor_idx,
                )
            self.__node_embeddings.append(node_embedding)
            self.__boundaris[worker_id] = boundary

    def __get_node_embedding(self, node_idx):
        list_idx, tensor_idx = self.__node_embedding_indices[node_idx]
        return self.__node_embeddings[list_idx][tensor_idx]

    def aggregate_worker_data(self) -> dict:
        res = super().aggregate_worker_data()
        if self.__node_embeddings:
            res["worker_result"] = {}
            for worker_id, boundary in self.__boundaris.items():
                new_boundary = [
                    node_idx
                    for node_idx in sorted(boundary)
                    if node_idx in self.__node_embedding_indices
                ]
                res["worker_result"][worker_id] = {
                    "boundary": new_boundary,
                    "node_embedding": torch.stack(
                        [
                            self.__get_node_embedding(node_idx)
                            for node_idx in new_boundary
                        ]
                    ),
                }
            self.__node_embeddings = []
            self.__node_embedding_indices = {}
            self.__boundaris = {}
        return res
