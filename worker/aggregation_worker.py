from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import ModelExecutorHookPoint
from cyy_torch_toolbox.model_util import ModelUtil
from cyy_torch_toolbox.tensor import tensor_to
from util.model import load_parameters


class AggregationWorker:
    def __init__(self, **kwargs):
        self._aggregation_time = ModelExecutorHookPoint.AFTER_EXECUTE
        self._reuse_learning_rate = False
        self._choose_model_by_validation: bool = True

    def _register_aggregation(self):
        get_logger().debug("use aggregation_time %s", self._aggregation_time)
        self.trainer.remove_hook(name="aggregation")
        self.trainer.append_named_hook(
            self._aggregation_time,
            "aggregation",
            self.__aggretation_impl,
        )

    def _should_aggregate(self, **kwargs):
        return True

    def __aggretation_impl(self, **kwargs):
        if not self._should_aggregate(**kwargs):
            return
        epoch = kwargs["epoch"]
        get_logger().debug("aggregation on epoch %s", epoch)

        sent_data = self._get_sent_data()
        self._aggretation(sent_data=sent_data)

    def _aggretation(self, sent_data):
        raise NotImplementedError()

    def _get_sent_data(self) -> dict:
        model_util = self.trainer.model_util
        if self._choose_model_by_validation:
            get_logger().debug("use best model")
            if self.trainer.best_model is not None:
                model_util = ModelUtil(self.trainer.best_model)
        return {"parameter": model_util.get_parameter_dict()}

    def _load_parameters(self, parameters):
        load_parameters(
            trainer=self.trainer,
            parameter_dict=parameters,
            reuse_learning_rate=self._reuse_learning_rate,
        )
