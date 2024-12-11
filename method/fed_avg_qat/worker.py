from cyy_torch_algorithm.quantization.qat import QuantizationAwareTraining
from distributed_learning_simulation import (
    AggregationWorker,
)


class QATWorker(AggregationWorker):
    def _before_training(self) -> None:
        super()._before_training()
        self.trainer.append_hook(QuantizationAwareTraining(), "QAT")
