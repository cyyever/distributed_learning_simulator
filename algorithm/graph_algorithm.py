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
        assert self.accumulate
        worker_data = self._all_worker_data[worker_id].data
        if worker_data is None:
            return
        if "node_embedding" in worker_data:
            node_embedding, node_indices = worker_data.pop("node_embedding")
            for tensor_idx, node_idx in enumerate(node_indices):
                assert node_idx not in self.__node_embedding_indices
                self.__node_embedding_indices[node_idx] = (
                    len(self.__node_embeddings),
                    tensor_idx,
                )
            self.__node_embeddings.append(node_embedding)
            self.__boundaris[worker_id] = worker_data.pop("boundary")

    def __get_node_embedding(self, node_idx):
        list_idx, tensor_idx = self.__node_embedding_indices[node_idx]
        return self.__node_embeddings[list_idx][tensor_idx]

    def aggregate_worker_data(self) -> dict:
        if "training_node_indices" in next(iter(self._all_worker_data.values())):
            training_node_indices = {}
            for worker_id, worker_data in self._all_worker_data.items():
                training_node_indices[worker_id] = worker_data["training_node_indices"]
            return {
                "training_node_indices": training_node_indices,
                "in_round_data": True,
            }
        res = super().aggregate_worker_data()

        if self.__node_embeddings:
            res["worker_result"] = {}
            node_embedding_index_set = set(self.__node_embedding_indices.keys())
            for worker_id, boundary in self.__boundaris.items():
                node_indices = boundary.intersection(node_embedding_index_set)
                node_indices = tuple(sorted(node_indices))
                res["worker_result"][worker_id] = {
                    "node_indices": node_indices,
                    "node_embedding": torch.stack(
                        [
                            self.__get_node_embedding(node_idx).cpu()
                            for node_idx in node_indices
                        ]
                    ),
                }
            self.__node_embeddings = []
            self.__node_embedding_indices = {}
            self.__boundaris = {}
            res["in_round_data"] = True
        return res
