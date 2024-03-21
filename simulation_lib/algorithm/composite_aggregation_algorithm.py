from typing import Any

from cyy_torch_toolbox.typing import TensorDict

from ..config import DistributedTrainingConfig
from ..message import Message
from .aggregation_algorithm import AggregationAlgorithm


class CompositeAggregationAlgorithm(AggregationAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.__algorithms: list[AggregationAlgorithm] = []
        self.__used_algorithm: AggregationAlgorithm | None = None

    def append_algorithm(self, algorithm: AggregationAlgorithm) -> None:
        self.__algorithms.append(algorithm)

    def set_old_parameter(self, old_parameter: TensorDict) -> None:
        for algorithm in self.__algorithms:
            algorithm.set_old_parameter(old_parameter=old_parameter)

    def set_config(self, config: DistributedTrainingConfig) -> None:
        for algorithm in self.__algorithms:
            algorithm.set_config(config=config)

    def process_worker_data(
        self,
        worker_id: int,
        worker_data: Message | None,
    ) -> bool:
        if self.__used_algorithm is not None:
            res = self.__used_algorithm.process_worker_data(
                worker_id=worker_id, worker_data=worker_data
            )
            assert res
            return True

        for algorithm in self.__algorithms:
            if algorithm.process_worker_data(
                worker_id=worker_id, worker_data=worker_data
            ):
                self.__used_algorithm = algorithm
                return True
        raise NotImplementedError("Failed to process_worker_data")

    def aggregate_worker_data(self) -> Any:
        assert self.__used_algorithm is not None
        res = self.__used_algorithm.aggregate_worker_data()
        self.__used_algorithm = None
        return res

    def clear_worker_data(self) -> None:
        for algorithm in self.__algorithms:
            algorithm.clear_worker_data()

    def exit(self) -> None:
        for algorithm in self.__algorithms:
            algorithm.exit()
