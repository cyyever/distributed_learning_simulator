from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.quantization.deterministic import (
    NNADQ, NeuralNetworkAdaptiveDeterministicDequant,
    NeuralNetworkAdaptiveDeterministicQuant)
from cyy_torch_algorithm.quantization.stochastic import stochastic_quantization

from .cs_endpoint import ClientEndpoint, ServerEndpoint


class QuantEndpoint:
    def __init__(self, quant, dequant):
        self._quant, self._dequant = quant, dequant


class QuantClientEndpoint(ClientEndpoint, QuantEndpoint):
    def __init__(self, quant, dequant, **kwargs):
        ClientEndpoint.__init__(self, **kwargs)
        QuantEndpoint.__init__(self, quant=quant, dequant=dequant)
        self.use_quantization: bool = True
        self.quantized_keys: None | set = None
        self.dequant_server_data: bool = False

    def get(self):
        data = super().get()
        if self.dequant_server_data:
            return self._dequant(data)
        return data

    def send(self, data):
        if data is None:
            super().send(data=data)
            return
        if self.use_quantization:
            if self.quantized_keys is not None:
                for k in self.quantized_keys:
                    if k in data:
                        data[k] = self._quant(data[k])
            else:
                data = self._quant(data)
            self._after_quant(data=data)
        else:
            get_logger().warning("stop quantization")
        super().send(data=data)

    def _after_quant(self, data):
        pass


class QuantServerEndpoint(ServerEndpoint, QuantEndpoint):
    def __init__(self, quant, dequant, **kwargs):
        ServerEndpoint.__init__(self, **kwargs)
        QuantEndpoint.__init__(self, quant=quant, dequant=dequant)
        self.quant_broadcast: bool = False
        self.client_quantized_keys: None | set = None

    def get(self, worker_id):
        data = super().get(worker_id=worker_id)
        if self.client_quantized_keys is None:
            return self._dequant(data)
        for k in self.client_quantized_keys:
            if k in data:
                data[k] = self._dequant(data[k])
        return data

    def broadcast(self, data, worker_ids=None):
        if self.quant_broadcast:
            data = self._quant(data)
            get_logger().warning("broadcast quantization")
            self._after_quant(data=data)
        else:
            get_logger().warning("stop quantization")
        return super().broadcast(data=data, worker_ids=worker_ids)

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

    def _after_quant(self, data):
        NeuralNetworkAdaptiveDeterministicQuant.check_compression_ratio(
            data, prefix="broadcast"
        )
