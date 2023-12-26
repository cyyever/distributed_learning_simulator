import os
from typing import Any

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.hook.keep_model import KeepModelHook
from cyy_torch_toolbox.ml_type import (ExecutorHookPoint, MachineLearningPhase,
                                       StopExecutingException)

from ..message import (DeltaParameterMessage, Message, ParameterMessage,
                       ParameterMessageBase)
from ..util.model import load_parameters
from ..util.model_cache import ModelCache
from .client import Client


class AggregationWorker(Client):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._aggregation_time: ExecutorHookPoint = ExecutorHookPoint.AFTER_EXECUTE
        self._reuse_learning_rate: bool = False
        self.__choose_model_by_validation: bool = False
        self._send_parameter_diff: bool = True
        self._model_cache: ModelCache = ModelCache()

    def _before_training(self) -> None:
        super()._before_training()
        self.trainer.dataset_collection.remove_dataset(phase=MachineLearningPhase.Test)
        if self.config.dataset_sampling == "iid":
            self.enable_choose_model_by_validation()
        if not self.__choose_model_by_validation:
            # Skip Validation to speed up training
            self.trainer.dataset_collection.remove_dataset(
                phase=MachineLearningPhase.Validation
            )
        # load initial parameters
        if self.config.distribute_init_parameters:
            self.__get_result_from_server()
            if self._stopped():
                return
        self._register_aggregation()

    def _register_aggregation(self) -> None:
        get_logger().debug("use aggregation_time %s", self._aggregation_time)
        self.trainer.remove_named_hook(name="aggregation")

        def __aggregation_impl(**kwargs) -> None:
            self._aggregation(sent_data=self._get_sent_data(), **kwargs)

        self.trainer.append_named_hook(
            self._aggregation_time,
            "aggregation",
            __aggregation_impl,
        )

    def _aggregation(self, sent_data: Message, **kwargs: Any) -> None:
        self.send_data_to_server(sent_data)
        self._offload_from_device()
        self.__get_result_from_server()

    def enable_choose_model_by_validation(self) -> None:
        self.__choose_model_by_validation = True
        hook = KeepModelHook()
        hook.keep_best_model = True
        assert self.trainer.dataset_collection.has_dataset(
            phase=MachineLearningPhase.Validation
        )
        self.trainer.append_hook(hook, "keep_model_hook")

    def disable_choose_model_by_validation(self) -> None:
        self.__choose_model_by_validation = False
        if self.best_model_hook is not None:
            self.trainer.remove_hook("keep_model_hook")

    @property
    def best_model_hook(self) -> KeepModelHook | None:
        if not self.trainer.has_hook_obj("keep_model_hook"):
            return None
        hook = self.trainer.get_hook("keep_model_hook")
        assert isinstance(hook, KeepModelHook)
        return hook

    def _get_sent_data(self) -> ParameterMessageBase:
        if self.__choose_model_by_validation:
            get_logger().debug("use best model")
            assert self.best_model_hook is not None
            parameter = self.best_model_hook.best_model["parameter"]
        else:
            parameter = self.trainer.model_util.get_parameter_dict()
        if self._send_parameter_diff:
            return DeltaParameterMessage(
                dataset_size=self.trainer.dataset_size,
                delta_parameter=self._model_cache.get_parameter_diff(parameter),
            )
        return ParameterMessage(
            dataset_size=self.trainer.dataset_size, parameter=parameter
        )

    def _load_result_from_server(self, result: Message) -> None:
        if result.end_training:
            self._force_stop = True
            raise StopExecutingException()
        model_path = os.path.join(
            self.config.save_dir, "aggregated_model", f"round_{self._round_num}.pk"
        )
        match result:
            case ParameterMessage():
                self._model_cache.cache_parameter_dict(
                    result.parameter, path=model_path
                )
            case DeltaParameterMessage():
                self._model_cache.add_parameter_diff(
                    result.delta_parameter, path=model_path
                )
            case _:
                raise NotImplementedError()
            # case ParameterFileMessage():
            #     self._model_cache.load_file(result.path)
        load_parameters(
            trainer=self.trainer,
            parameter_dict=self._model_cache.parameter_dict,
            reuse_learning_rate=self._reuse_learning_rate,
        )

    def _offload_from_device(self) -> None:
        if self.config.limited_resource:
            self._model_cache.save()

        if self.best_model_hook is not None:
            self.best_model_hook.clear()
        super()._offload_from_device()

    def __get_result_from_server(self) -> None:
        while True:
            result = super()._get_data_from_server()
            get_logger().debug("get result from server %s", type(result))
            if result is None:
                get_logger().debug("skip round %s", self._round_num)
                self._round_num += 1
                self.send_data_to_server(None)
                if self._stopped():
                    return
            self._load_result_from_server(result=result)
            break
        return
