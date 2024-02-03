from typing import Any

from cyy_torch_toolbox.typing import TensorDict

from ..config import DistributedTrainingConfig
from ..message import Message, ParameterMessage


class AggregationAlgorithm:
    def __init__(self) -> None:
        self._all_worker_data: dict[int, Message] = {}
        self.__skipped_workers: set[int] = set()
        self._old_parameter: TensorDict | None = None
        self._config: DistributedTrainingConfig | None = None

    def set_old_parameter(self, old_parameter: TensorDict) -> None:
        self._old_parameter = old_parameter

    def set_config(self, config: DistributedTrainingConfig) -> None:
        self._config = config

    @classmethod
    def get_ratios(cls, data_dict: dict[int, ParameterMessage]) -> dict[int, float]:
        total_scalar = sum(v.aggregation_weight for v in data_dict.values())
        return {
            k: float(v.aggregation_weight) / float(total_scalar)
            for k, v in data_dict.items()
        }

    @classmethod
    def weighted_avg(
        cls,
        data_dict: dict[int, ParameterMessage],
        weights: dict[int, float] | float,
    ) -> TensorDict:
        assert data_dict
        avg_data: TensorDict = {}
        for worker_id, v in data_dict.items():
            if isinstance(weights, dict):
                weight = weights[worker_id]
            else:
                weight = weights
            assert 0 <= weight <= 1

            d = {k2: v2 * weight for (k2, v2) in v.parameter.items()}
            if not avg_data:
                avg_data = d
            else:
                for k in avg_data:
                    avg_data[k] += d[k]
        for p in avg_data.values():
            assert not p.isnan().any().cpu()
        return avg_data

    def process_worker_data(
        self,
        worker_id: int,
        worker_data: Message | None,
    ) -> None:
        if worker_data is None:
            self.__skipped_workers.add(worker_id)
            return
        self._all_worker_data[worker_id] = worker_data

    def aggregate_worker_data(self) -> Any:
        raise NotImplementedError()

    def clear_worker_data(self) -> None:
        self._all_worker_data.clear()
        self.__skipped_workers.clear()

    def exit(self) -> None:
        pass
