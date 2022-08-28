from cyy_torch_algorithm.quantization.stochastic import stochastic_quantization

from .gradient_worker import GradientWorker


class QSGDWorker(GradientWorker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quantization_level = 256
        self.quant, self.dequant = stochastic_quantization(self.quantization_level)

    def _process_gradient(self, gradient):
        quantized_pair = self.quant(gradient)
        self.send_data_to_server(
            {
                "dataset_size": self.trainer.dataset_size,
                "quantized_pair": quantized_pair,
                "dequant": self.dequant,
            }
        )
        return self.get_result_from_server()
