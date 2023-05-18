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
        self._round_number = 1
        self._send_parameter_path = False
        self.__worker_flag: set = set()
        self.__algorithm: AggregationAlgorithm = algorithm
        self.__init_global_model_path = self.config.algorithm_kwargs.get(
            "global_model_path", None
        )

    @property
    def algorithm(self):
        return self.__algorithm

    @property
    def round_number(self):
        return self._round_number

    def _distribute_init_model(self):
        with self._get_context():
            if self.config.distribute_init_parameters:
                self.send_result(
                    self.__algorithm.process_init_model(self.__get_init_model())
                )

    def __get_init_model(self) -> dict:
        parameter_dict: dict = {}
        if self.__init_global_model_path is not None:
            with open(os.path.join(self.__init_global_model_path), "rb") as f:
                parameter_dict = pickle.load(f)
        else:
            parameter_dict = self.tester.model_util.get_parameter_dict()
            # save GPU memory
            self.tester.offload_from_gpu()
        return parameter_dict

    def start(self) -> None:
        self._distribute_init_model()
        super().start()

    def _server_exit(self) -> None:
        self.__algorithm.exit()

    def _process_worker_data(self, worker_id, data):
        assert 0 <= worker_id < self.worker_number
        get_logger().debug("get data %s from worker %s", data, worker_id)
        self.__algorithm.process_worker_data(
            worker_id=worker_id,
            worker_data=data,
            save_dir=self.save_dir,
            old_parameter_dict=self._model_cache.get_parameter_dict,
        )
        self.__worker_flag.add(worker_id)
        if len(self.__worker_flag) == self.worker_number:
            result = self._aggregate_worker_data()
            self._after_aggregate_worker_data(result)
        else:
            get_logger().debug(
                "we have %s committed, and we need %s workers,skip",
                len(self.__worker_flag),
                self.worker_number,
            )

    def _aggregate_worker_data(self) -> dict:
        return self.__algorithm.aggregate_worker_data()

    def _after_aggregate_worker_data(self, result) -> None:
        self.send_result(result)
        if "in_round_data" not in result:
            self._round_number += 1
        self.__worker_flag.clear()

    def send_result(self, result) -> None:
        parameter = result.pop("parameter", None)
        if parameter is not None:
            model_path = os.path.join(
                self.save_dir, "aggregated_model", f"round_{self.round_number}.pk"
            )
            self._model_cache.cache_parameter_dict(parameter, model_path)
            if "partial_parameter" not in result:
                if self._send_parameter_path:
                    result["parameter_path"] = self._model_cache.get_parameter_path()
                else:
                    result["parameter"] = parameter
        super().send_result(result)

    def _stopped(self) -> bool:
        return self._round_number > self.config.round
