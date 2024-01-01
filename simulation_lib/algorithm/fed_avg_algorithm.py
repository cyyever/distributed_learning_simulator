from typing import Any

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.typing import TensorDict

from ..message import Message, ParameterMessage
from .aggregation_algorithm import AggregationAlgorithm


class FedAVGAlgorithm(AggregationAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.accumulate: bool = True
        self.__dtypes: dict[str, Any] = {}
        self.__total_weights: dict[str, float] = {}
        self.__parameter: TensorDict = {}

    def process_worker_data(
        self,
        worker_id: int,
        worker_data: Message | None,
        old_parameter_dict: TensorDict | None,
        save_dir: str,
    ) -> None:
        super().process_worker_data(
            worker_id=worker_id,
            worker_data=worker_data,
            old_parameter_dict=old_parameter_dict,
            save_dir=save_dir,
        )
        if not self.accumulate:
            return
        worker_data = self._all_worker_data.get(worker_id, None)
        if worker_data is None:
            return
        if not isinstance(worker_data, ParameterMessage):
            return
        for k, v in worker_data.parameter.items():
            self.__dtypes[k] = v.dtype
            weight = self._get_weight(
                dataset_size=worker_data.dataset_size, name=k, parameter=v
            )
            tmp = v.to(dtype=torch.float64) * weight
            if k not in self.__parameter:
                self.__parameter[k] = tmp
            else:
                self.__parameter[k] += tmp
            if k not in self.__total_weights:
                self.__total_weights[k] = weight
            else:
                self.__total_weights[k] += weight
        # release to reduce memory pressure
        worker_data.parameter = {}

    def _get_weight(self, dataset_size: int, name: str, parameter: Any) -> Any:
        assert dataset_size != 0
        return dataset_size

    def _apply_total_weight(
        self, name: str, parameter: torch.Tensor, total_weight: Any
    ) -> torch.Tensor:
        return parameter / total_weight

    def aggregate_worker_data(self) -> Message:
        if not self.accumulate:
            parameter = self._aggregate_worker_data(self._all_worker_data)
        else:
            assert self.__parameter
            parameter = self.__parameter
            self.__parameter = {}
            for k, v in parameter.items():
                parameter[k] = self._apply_total_weight(
                    name=k, parameter=v, total_weight=self.__total_weights[k]
                ).to(dtype=self.__dtypes[k])
                assert not parameter[k].isnan().any().cpu()
            self.__total_weights = {}
        return ParameterMessage(
            parameter=parameter,
            end_training=next(iter(self._all_worker_data.values())).end_training,
            in_round=next(iter(self._all_worker_data.values())).in_round,
            other_data=self.__check_and_reduce_other_data(self._all_worker_data),
        )

    @classmethod
    def _aggregate_worker_data(
        cls, all_worker_data: dict[int, ParameterMessage]
    ) -> TensorDict:
        assert all_worker_data
        assert isinstance(next(iter(all_worker_data.values())), ParameterMessage)
        parameter = AggregationAlgorithm.weighted_avg(
            all_worker_data,
            AggregationAlgorithm.get_ratios(all_worker_data),
        )
        assert parameter
        return parameter

    @classmethod
    def __check_and_reduce_other_data(cls, all_worker_data: dict) -> dict:
        result: dict = {}
        for worker_data in all_worker_data.values():
            for k, v in worker_data.other_data.items():
                if k not in result:
                    result[k] = v
                    continue
                if v != result[k]:
                    get_logger().error("different values on key %s", k)
                    raise RuntimeError(f"different values on key {k}")
        return result
