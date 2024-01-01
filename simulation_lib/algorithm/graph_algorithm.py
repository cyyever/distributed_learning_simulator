import torch
from cyy_naive_lib.log import get_logger

from ..message import Message, ParameterMessageBase
from .fed_avg_algorithm import FedAVGAlgorithm


class GraphNodeEmbeddingPassingAlgorithm(FedAVGAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__node_embeddings: list[torch.Tensor] = []
        self.__node_embedding_indices: dict = {}
        self.__boundaris: dict = {}
        self.__training_node_indices: dict = {}

    def process_worker_data(
        self,
        worker_id: int,
        worker_data: Message | None,
        old_parameter_dict: dict | None,
        save_dir: str,
    ) -> None:
        if worker_data is not None and worker_data.in_round:
            assert not isinstance(worker_data, ParameterMessageBase)
            if "training_node_indices" in worker_data.other_data:
                self.__training_node_indices[worker_id] = worker_data.other_data.pop(
                    "training_node_indices"
                )
            if "node_embedding" in worker_data.other_data:
                node_embedding_tuple = worker_data.other_data.pop("node_embedding")
                if node_embedding_tuple is not None:
                    node_embedding, node_indices = node_embedding_tuple
                    for tensor_idx, node_idx in enumerate(node_indices):
                        assert node_idx not in self.__node_embedding_indices
                        self.__node_embedding_indices[node_idx] = (
                            len(self.__node_embeddings),
                            tensor_idx,
                        )
                    self.__node_embeddings.append(node_embedding)
                self.__boundaris[worker_id] = worker_data.other_data.pop("boundary")
            assert not worker_data.other_data
        else:
            super().process_worker_data(
                worker_id=worker_id,
                worker_data=worker_data,
                old_parameter_dict=old_parameter_dict,
                save_dir=save_dir,
            )

    def __get_node_embedding(self, node_idx):
        list_idx, tensor_idx = self.__node_embedding_indices[node_idx]
        return self.__node_embeddings[list_idx][tensor_idx]

    def aggregate_worker_data(self) -> Message:
        if self.__training_node_indices:
            msg = Message(
                in_round=True,
                other_data={"training_node_indices": self.__training_node_indices},
            )
            self.__training_node_indices = {}
            return msg

        if self.__node_embeddings:
            # node_embedding_size = 0
            # for data in self.__node_embeddings:
            #     node_embedding_size += data.element_size() * data.numel()
            # get_logger().error("node_embedding_size %s", node_embedding_size)
            res: dict = {}
            res["worker_result"] = {}
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
                res["worker_result"][worker_id] = {
                    "node_indices": node_indices,
                    "node_embedding": node_embedding,
                }
            self.__node_embeddings = []
            self.__node_embedding_indices = {}
            self.__boundaris = {}
            return Message(in_round=True, other_data=res)
        return super().aggregate_worker_data()
