from typing import Any

import torch
from cyy_torch_toolbox.typing import TensorDict

from ..message import ParameterMessageBase
from .aggregation_worker import AggregationWorker


class ErrorFeedbackWorker(AggregationWorker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert self._send_parameter_diff
        self.__error: TensorDict = {}

    def _get_sent_data(self) -> ParameterMessageBase:
        return self._sparsify(super()._get_sent_data())

    def _sparsify(self, sent_data: ParameterMessageBase) -> ParameterMessageBase:
        raise NotImplementedError()

    def _get_error(self, name: str, param: torch.Tensor) -> torch.Tensor:
        if name not in self.__error:
            self.__error[name] = torch.zeros_like(param)
        return self.__error[name]

    def _set_error(self, name: str, error: torch.Tensor) -> None:
        self.__error[name] = error
