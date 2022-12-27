from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import ModelExecutorHookPoint
from cyy_torch_toolbox.model_util import ModelUtil
from util.model import load_parameters
from util.model_cache import ModelCache


class AggregationWorker:
    def __init__(self, trainer, **kwargs):
        self._aggregation_time = ModelExecutorHookPoint.AFTER_EXECUTE
        self._reuse_learning_rate = False
        self._choose_model_by_validation: bool = True
        self._trainer = trainer
        self._send_parameter_diff = True
        self._model_cache: ModelCache = ModelCache()

    def _register_aggregation(self):
        get_logger().debug("use aggregation_time %s", self._aggregation_time)
        self._trainer.remove_named_hook(name="aggregation")
        self._trainer.append_named_hook(
            self._aggregation_time,
            "aggregation",
            self.__aggregation_impl,
        )

    def _should_aggregate(self, **kwargs):
        return True

    def __aggregation_impl(self, **kwargs):
        if not self._should_aggregate(**kwargs):
            return
        sent_data = self._get_sent_data()
        self._aggregation(sent_data=sent_data)

    def _aggregation(self, sent_data):
        raise NotImplementedError()

    def _get_sent_data(self) -> dict:
        model_util = self._trainer.model_util
        if self._choose_model_by_validation:
            get_logger().debug("use best model")
            if self._trainer.best_model is not None:
                model_util = ModelUtil(self._trainer.best_model)

        sent_data = {
            "dataset_size": self._trainer.dataset_size,
        }
        parameter = model_util.get_parameter_dict()
        if self._send_parameter_diff:
            sent_data["parameter_diff"] = self._model_cache.get_parameter_diff(
                parameter
            )
        else:
            sent_data["parameter"] = parameter
        return sent_data

    def _load_parameters(self, parameters):
        load_parameters(
            trainer=self._trainer,
            parameter_dict=parameters,
            reuse_learning_rate=self._reuse_learning_rate,
        )
        self._model_cache.cache_parameter_dict(
            self._trainer.model_util.get_parameter_dict()
        )
