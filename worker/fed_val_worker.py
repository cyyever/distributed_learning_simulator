from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import (MachineLearningPhase,
                                       ModelExecutorHookPoint)

from .fed_avg_worker import FedAVGWorker


class FedValidationWorker(FedAVGWorker):
    __validation_loss = None
    _use_validation_loss: bool = True

    def _before_training(self):
        super()._before_training()
        self.trainer.append_named_hook(
            ModelExecutorHookPoint.BEFORE_EXECUTE,
            "compute_val_loss",
            self._compute_val_loss,
        )

    def _get_sent_data(self):
        assert self.__validation_loss is not None
        sent_data = super()._get_sent_data()
        if not self._use_validation_loss:
            return sent_data
        parameter_dict = sent_data["parameter"]
        cur_loss = self._get_validation_loss(parameter_dict)
        if cur_loss >= self.__validation_loss:
            get_logger().warning(
                "use_distributed_model, cur_loss is %s, round begin loss is %s",
                cur_loss,
                self.__validation_loss,
            )
            self.__validation_loss = None
            return {"use_distributed_model": True}
        get_logger().debug("loss reduction is %s", cur_loss - self.__validation_loss)
        self.__validation_loss = None
        return sent_data

    def _get_validation_loss(self, parameter_dict=None):
        if parameter_dict is not None:
            inferencer = self.trainer.get_inferencer(
                MachineLearningPhase.Validation, copy_model=True
            )
            inferencer.model_util.load_parameter_dict(parameter_dict)
        else:
            inferencer = self.trainer.get_inferencer(
                MachineLearningPhase.Validation, copy_model=False
            )
        inferencer.disable_logger()
        inferencer.disable_performance_metric_logger()
        inferencer.inference(epoch=1)
        return inferencer.performance_metric.get_loss(1).item()

    def _compute_val_loss(self, **kwargs):
        self.__validation_loss = self._get_validation_loss()
