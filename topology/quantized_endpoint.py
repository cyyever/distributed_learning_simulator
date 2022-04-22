from cyy_torch_algorithm.quantization.stochastic import stochastic_quantization


class QuantizedEndpoint:
    quant, dequant = stochastic_quantization(quantization_level=255)
