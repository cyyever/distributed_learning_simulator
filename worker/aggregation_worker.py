import copy
import os

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import ModelExecutorHookPoint
from cyy_torch_toolbox.model_util import ModelUtil
from cyy_torch_toolbox.tensor import (cat_tensors_to_vector,
                                      get_tensor_serialization_size)
from util import load_parameters


class AggregationWorker:
    def __init__(self, aggregation_time=ModelExecutorHookPoint.AFTER_EXECUTE, **kwargs):
        super().__init__(**kwargs)
        self.__aggregation_time = aggregation_time
        self.__compute_transfer_rate: bool = False
        get_logger().debug("use aggregation_time %s", self.__aggregation_time)

    def _register_aggregation(self):
        self.trainer.append_named_hook(
            self.__aggregation_time,
            "aggregation",
            self.__aggretation_impl,
        )

    def __aggretation_impl(self, **kwargs):
        trainer = kwargs["model_executor"]
        epoch = kwargs["epoch"]
        get_logger().warning("aggregation on epoch %s", epoch)

        parameter_data = self._get_parameter_data()
        if self.__compute_transfer_rate:
            if parameter_data is None:
                transferred_size = 0
            else:
                if isinstance(parameter_data, dict):
                    transferred_size = get_tensor_serialization_size(
                        cat_tensors_to_vector(parameter_data.values())
                    )
                elif isinstance(parameter_data, tuple):
                    transferred_size = get_tensor_serialization_size(parameter_data[0])

            parameter_size = get_tensor_serialization_size(
                cat_tensors_to_vector(
                    ModelUtil(self.trainer.model).get_parameter_dict().values()
                )
            )
            get_logger().info(
                "transferred size %s parameter_size %s compression ratio %s",
                transferred_size,
                parameter_size,
                transferred_size / parameter_size,
            )
        self._aggretation(trainer=trainer, parameter_data=parameter_data)

    def _aggretation(self, trainer, parameter_data):
        raise NotImplementedError()

    def _get_parameter_data(self):
        if os.getenv("use_best_model", default="1") == "1":
            get_logger().info("use best model")
            return copy.deepcopy(
                ModelUtil(self.trainer.best_model).get_parameter_dict()
            )
        return copy.deepcopy(self.trainer.model_util.get_parameter_dict())

    def _load_parameters(self, parameters, reuse_learning_rate):
        load_parameters(
            trainer=self.trainer,
            parameter_dict=parameters,
            reuse_learning_rate=reuse_learning_rate,
        )
        self.trainer.exec_hooks(ModelExecutorHookPoint.AFTER_LOAD_MODEL)
