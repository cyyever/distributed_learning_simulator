from typing import Any

from cyy_torch_toolbox.typing import TensorDict

from ..message import ParameterMessageBase
from .aggregation_worker import AggregationWorker


class ErrorFeedbackWorker(AggregationWorker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert self._send_parameter_diff
        self._error: TensorDict = {}

    def _get_sent_data(self) -> ParameterMessageBase:
        return self.sparsify(super()._get_sent_data())

    def sparsify(self, sent_data: ParameterMessageBase) -> ParameterMessageBase:
        raise NotImplementedError()
