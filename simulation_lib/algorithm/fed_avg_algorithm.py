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
        self.aggregate_loss: bool = False
        self.__dtypes: dict[str, Any] = {}
        self.__total_weights: dict[str, float] = {}
        self.__parameter: TensorDict = {}

    def process_worker_data(
        self,
        worker_id: int,
        worker_data: Message | None,
    ) -> bool:
        res = super().process_worker_data(worker_id=worker_id, worker_data=worker_data)
        if not res:
            return False
        if not self.accumulate:
            return True
        worker_data = self._all_worker_data.get(worker_id, None)
        if worker_data is None:
            return True
        if not isinstance(worker_data, ParameterMessage):
            return True
        for k, v in worker_data.parameter.items():
            assert not v.isnan().any().cpu()
            self.__dtypes[k] = v.dtype
            weight = self._get_weight(worker_data, name=k, parameter=v)
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
        return True

    def _get_weight(
        self, worker_data: ParameterMessage, name: str, parameter: Any
    ) -> Any:
        return worker_data.aggregation_weight

    def _apply_total_weight(
        self, name: str, parameter: torch.Tensor, total_weight: Any
    ) -> torch.Tensor:
        return parameter / total_weight

    def aggregate_worker_data(self) -> ParameterMessage:
        if not self.accumulate:
            parameter = self.aggregate_parameter(self._all_worker_data)
        else:
            assert self.__parameter
            parameter = self.__parameter
            self.__parameter = {}
            for k, v in parameter.items():
                assert not v.isnan().any().cpu()
                parameter[k] = self._apply_total_weight(
                    name=k, parameter=v, total_weight=self.__total_weights[k]
                ).to(dtype=self.__dtypes[k])
                assert not parameter[k].isnan().any().cpu()
            self.__total_weights = {}
        other_data: dict[str, Any] = {}
        if self.aggregate_loss:
            other_data |= self.__aggregate_loss(self._all_worker_data)
        other_data |= self.__check_and_reduce_other_data(self._all_worker_data)
        return ParameterMessage(
            parameter=parameter,
            end_training=next(iter(self._all_worker_data.values())).end_training,
            in_round=next(iter(self._all_worker_data.values())).in_round,
            other_data=other_data,
        )

    @classmethod
    def aggregate_parameter(cls, all_worker_data: dict[int, Message]) -> TensorDict:
        assert all_worker_data
        assert all(
            isinstance(parameter, ParameterMessage)
            for parameter in all_worker_data.values()
        )
        parameter = AggregationAlgorithm.weighted_avg(
            all_worker_data,
            AggregationAlgorithm.get_ratios(all_worker_data),
        )
        assert parameter
        return parameter

    @classmethod
    def __aggregate_loss(cls, all_worker_data: dict[int, Message]) -> dict:
        assert all_worker_data
        loss_dict = {}
        for worker_data in all_worker_data.values():
            for loss_type in ("training_loss", "validation_loss"):
                if loss_type in worker_data.other_data:
                    loss_dict[loss_type] = AggregationAlgorithm.weighted_avg_for_scalar(
                        all_worker_data,
                        AggregationAlgorithm.get_ratios(all_worker_data),
                        scalar_key=loss_type,
                    )
            break
        assert loss_dict
        for worker_data in all_worker_data.values():
            for loss_type in ("training_loss", "validation_loss"):
                worker_data.other_data.pop(loss_type, None)
        return loss_dict

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
