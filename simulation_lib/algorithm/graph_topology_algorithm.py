from ..message import Message
from .aggregation_algorithm import AggregationAlgorithm


class GraphTopologyAlgorithm(AggregationAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self._training_node_indices: dict = {}

    def process_worker_data(self, worker_id: int, worker_data: Message | None) -> bool:
        if (
            worker_data is not None
            and "training_node_indices" in worker_data.other_data
        ):
            self._training_node_indices[worker_id] = worker_data.other_data.pop(
                "training_node_indices"
            )
            assert not worker_data.other_data
            return True
        return False

    def aggregate_worker_data(self) -> Message:
        return Message(
            in_round=True,
            other_data={"training_node_indices": self._training_node_indices},
        )
