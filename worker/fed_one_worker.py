from cyy_torch_toolbox.ml_type import ModelExecutorHookPoint

from .fed_avg_worker import FedAVGWorker


class FedOneWorker(FedAVGWorker):
    def __init__(self, **kwargs):
        super().__init__(
            aggregation_time=ModelExecutorHookPoint.AFTER_VALIDATION,
            reuse_learning_rate=True,
            **kwargs
        )
        assert self.config.round == 1
