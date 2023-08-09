from typing import Any, Callable

from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.quantization.deterministic import (
    NNADQ, NeuralNetworkAdaptiveDeterministicDequant,
    NeuralNetworkAdaptiveDeterministicQuant)
from cyy_torch_algorithm.quantization.stochastic import stochastic_quantization

from .cs_endpoint import ClientEndpoint, ServerEndpoint


class QuantClientEndpoint(ClientEndpoint):
    def __init__(self, quant: Callable, dequant: Callable, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._quant: Callable = quant
        self._dequant: Callable = dequant
        self.quantized_keys: set = set(["parameter", "parameter_diff"])
        self.dequant_server_data: bool = False

    def get(self):
        data = super().get()
        if data is None:
            return data
        if self.dequant_server_data:
            for key in self.quantized_keys:
                if key in data:
                    data[key] = self._dequant(data[key])
            return self._dequant(data)
        return data

    def send(self, data):
        if data is None:
            super().send(data=data)
            return
        if self.quantized_keys:
            get_logger().debug("before client quantization")
            has_quantized = False
            assert self.quantized_keys
            for k in self.quantized_keys:
                if k in data:
                    data[k] = self._quant(data[k])
                    has_quantized = True
            assert has_quantized
            self._after_quant(data=data)
            get_logger().debug("after client quantization")
        else:
            get_logger().warning("client not use quantization")
        super().send(data=data)

    def _after_quant(self, data):
        pass


class QuantServerEndpoint(ServerEndpoint):
    def __init__(self, quant: Callable, dequant: Callable, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._quant: Callable = quant
        self._dequant: Callable = dequant
        self.quant_broadcast: bool = False
        self.client_quantized_keys: set = set(["parameter", "parameter_diff"])
        self.quantized_keys: set = set(["parameter", "parameter_diff"])

    def get(self, worker_id):
        data = super().get(worker_id=worker_id)
        if data is None:
            return data
        for k in self.client_quantized_keys:
            if k in data:
                data[k] = self._dequant(data[k])
        return data

    def broadcast(self, data, **kwargs):
        if data is not None:
            if self.quant_broadcast:
                has_quantized = False
                for k in data:
                    if k in self.quantized_keys:
                        data[k] = self._quant(data[k])
                        has_quantized = True
                assert has_quantized
                get_logger().warning("broadcast quantization")
                self._after_quant(data=data)
            else:
                get_logger().warning("server not use quantization")
        return super().broadcast(data=data, **kwargs)

    def _after_quant(self, data):
        pass


class StochasticQuantClientEndpoint(QuantClientEndpoint):
    def __init__(self, **kwargs):
        quant, dequant = stochastic_quantization(quantization_level=255)
        super().__init__(quant=quant, dequant=dequant, **kwargs)


class StochasticQuantServerEndpoint(QuantServerEndpoint):
    def __init__(self, **kwargs):
        quant, dequant = stochastic_quantization(quantization_level=255)
        super().__init__(quant=quant, dequant=dequant, **kwargs)


class NNADQClientEndpoint(QuantClientEndpoint):
    def __init__(self, weight, **kwargs):
        get_logger().info("use weight %s", weight)
        quant, dequant = NNADQ(weight=weight)
        super().__init__(quant=quant, dequant=dequant, **kwargs)

    def _after_quant(self, data):
        NeuralNetworkAdaptiveDeterministicQuant.check_compression_ratio(
            quantized_data=data, prefix="worker"
        )


class NNADQServerEndpoint(QuantServerEndpoint):
    def __init__(self, weight=None, **kwargs):
        if weight is None:
            quant = None
            dequant = NeuralNetworkAdaptiveDeterministicDequant()
        else:
            quant, dequant = NNADQ(weight=weight)
        super().__init__(quant=quant, dequant=dequant, **kwargs)
        self.__quant_callback = None

    def set_quant_callback(self, quant_callback):
        self.__quant_callback = quant_callback

    def _after_quant(self, data):
        NeuralNetworkAdaptiveDeterministicQuant.check_compression_ratio(
            data, prefix="broadcast"
        )
        if self.__quant_callback is not None:
            self.__quant_callback(data=data)
