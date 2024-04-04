from typing import Any, Callable

from cyy_naive_lib.log import get_logger
from cyy_naive_lib.topology.cs_endpoint import ClientEndpoint, ServerEndpoint
from cyy_torch_algorithm.quantization.deterministic import (
    NNADQ, NeuralNetworkAdaptiveDeterministicDequant,
    NeuralNetworkAdaptiveDeterministicQuant)
from cyy_torch_algorithm.quantization.stochastic import stochastic_quantization

from ..message import (DeltaParameterMessage, ParameterMessage,
                       ParameterMessageBase)


class QuantClientEndpoint(ClientEndpoint):
    def __init__(self, quant: Callable, dequant: Callable, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._quant: Callable = quant
        self._dequant: Callable = dequant
        self.dequant_server_data: bool = False

    def get(self) -> Any:
        data = super().get()
        if not self.dequant_server_data or data is None:
            return data
        assert isinstance(data, ParameterMessage)
        data.parameter = self._dequant(data.parameter)
        return self._dequant(data)

    def send(self, data: Any) -> None:
        if isinstance(data, ParameterMessage):
            data.parameter = self._quant(data.parameter)
            self._after_quant(data=data)
            get_logger().debug("after client quantization")
        super().send(data=data)

    def _after_quant(self, data: Any) -> None:
        pass


class QuantServerEndpoint(ServerEndpoint):
    def __init__(
        self, quant: Callable | None, dequant: Callable, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._quant: Callable | None = quant
        self._dequant: Callable = dequant
        self.use_quant: bool = False

    def get(self, worker_id) -> Any:
        data = super().get(worker_id=worker_id)
        match data:
            case ParameterMessage():
                data.parameter = self._dequant(data.parameter)
            case DeltaParameterMessage():
                data.delta_parameter = self._dequant(data.delta_parameter)
        return data

    def send(self, worker_id: int, data: Any) -> None:
        if isinstance(data, ParameterMessageBase):
            quantized: bool = data.other_data.pop("quantized", False)
            if not quantized:
                if self.use_quant:
                    assert isinstance(data, ParameterMessage)
                    assert self._quant is not None
                    data.parameter = self._quant(data.parameter)
                    data.other_data["quantized"] = True
                    get_logger().debug("call after_quant for worker %s", worker_id)
                    self._after_quant(data=data)
                    get_logger().debug(
                        "after_quant quantization for worker %s", worker_id
                    )
                else:
                    get_logger().debug("server not use quantization")
        get_logger().debug("before send quantized data to worker %s", worker_id)
        super().send(worker_id=worker_id, data=data)
        get_logger().debug("after send quantized data to worker %s", worker_id)

    def _after_quant(self, data: Any) -> None:
        pass


class StochasticQuantClientEndpoint(QuantClientEndpoint):
    def __init__(self, **kwargs: Any) -> None:
        quant, dequant = stochastic_quantization(quantization_level=255)
        super().__init__(quant=quant, dequant=dequant, **kwargs)


class StochasticQuantServerEndpoint(QuantServerEndpoint):
    def __init__(self, **kwargs: Any) -> None:
        quant, dequant = stochastic_quantization(quantization_level=255)
        super().__init__(quant=quant, dequant=dequant, **kwargs)


class NNADQClientEndpoint(QuantClientEndpoint):
    def __init__(self, weight: float, **kwargs: Any) -> None:
        get_logger().debug("use weight %s", weight)
        quant, dequant = NNADQ(weight=weight)
        super().__init__(quant=quant, dequant=dequant, **kwargs)

    def _after_quant(self, data: Any) -> None:
        NeuralNetworkAdaptiveDeterministicQuant.check_compression_ratio(
            quantized_data=data, prefix="worker"
        )


class NNADQServerEndpoint(QuantServerEndpoint):
    def __init__(self, weight: float | None = None, **kwargs: Any) -> None:
        if weight is None:
            quant = None
            dequant = NeuralNetworkAdaptiveDeterministicDequant()
        else:
            quant, dequant = NNADQ(weight=weight)
        super().__init__(quant=quant, dequant=dequant, **kwargs)

    def _after_quant(self, data: Any) -> None:
        NeuralNetworkAdaptiveDeterministicQuant.check_compression_ratio(
            data, prefix="broadcast"
        )
