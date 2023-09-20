import os
import pickle
from typing import Any

from algorithm.aggregation_algorithm import AggregationAlgorithm
from cyy_naive_lib.log import get_logger
from util.model_cache import ModelCache

from .server import Server


class AggregationServer(Server):
    def __init__(
        self, algorithm: AggregationAlgorithm, *args: Any, **kwargs: Any
    ) -> None:
        Server.__init__(self, *args, **kwargs)
        self._model_cache: ModelCache = ModelCache()
        self._round_number: int = 1
        self.__worker_flag: set = set()
        self.__algorithm: AggregationAlgorithm = algorithm

    @property
    def algorithm(self):
        return self.__algorithm

    @property
    def round_number(self):
        return self._round_number

    def _get_init_model(self) -> dict:
        parameter_dict: dict = {}
        init_global_model_path = self.config.algorithm_kwargs.get(
            "global_model_path", None
        )
        if init_global_model_path is not None:
            with open(os.path.join(init_global_model_path), "rb") as f:
                parameter_dict = pickle.load(f)
        else:
            parameter_dict = self.tester.model_util.get_parameter_dict()
            # save GPU memory
            self.tester.offload_from_device()
        return parameter_dict

    def _before_start(self) -> None:
        if self.config.distribute_init_parameters:
            self._send_result(
                self.__algorithm.process_init_model(self._get_init_model())
            )

    def _server_exit(self) -> None:
        self.__algorithm.exit()

    def _process_worker_data(self, worker_id: int, data: dict[str, Any]) -> None:
        assert 0 <= worker_id < self.worker_number
        get_logger().debug("get data %s from worker %s", data, worker_id)
        self.__algorithm.process_worker_data(
            worker_id=worker_id,
            worker_data=data,
            save_dir=self.config.get_save_dir(),
            old_parameter_dict=self._model_cache.parameter_dict,
        )
        self.__worker_flag.add(worker_id)
        if len(self.__worker_flag) == self.worker_number:
            result = self._aggregate_worker_data()
            self._send_result(result)
            self.__worker_flag.clear()
        else:
            get_logger().debug(
                "we have %s committed, and we need %s workers,skip",
                len(self.__worker_flag),
                self.worker_number,
            )

    def _aggregate_worker_data(self) -> dict:
        return self.__algorithm.aggregate_worker_data()

    def _before_send_result(self, result: dict) -> None:
        parameter: dict | None = result.pop("parameter", None)
        if parameter is None:
            return
        model_path = os.path.join(
            self.config.save_dir,
            "aggregated_model",
            f"round_{self.round_number}.pk",
        )
        self._model_cache.cache_parameter_dict(parameter, model_path)
        if "partial_parameter" in result:
            return
        if self.config.limited_resource:
            result["parameter_path"] = self._model_cache.get_parameter_path()
        else:
            result["parameter"] = parameter

    def _after_send_result(self, result: dict) -> None:
        if "in_round_data" not in result and "init_parameter" not in result:
            self._round_number += 1
            print("add round number", self._round_number)
        self.__algorithm.clear_worker_data()

    def _stopped(self) -> bool:
        return self._round_number > self.config.round
