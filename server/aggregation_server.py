import os
import pickle

from cyy_naive_lib.log import get_logger
from cyy_naive_lib.storage import DataStorage
from cyy_torch_toolbox.device import get_cpu_device
from cyy_torch_toolbox.tensor import tensor_to
from util.model_cache import ModelCache

from .server import Server


class AggregationServer(Server, ModelCache):
    def __init__(self, algorithm, *args, **kwargs):
        Server.__init__(self, *args, **kwargs)
        ModelCache.__init__(self)
        self._round_number = 0
        self._send_parameter_diff = False
        self.__worker_data: dict[int, DataStorage | None] = {}
        self.__algorithm = algorithm
        self.__init_global_model_path = self.config.algorithm_kwargs.get(
            "global_model_path", None
        )

    @property
    def round_number(self):
        return self._round_number

    def _distribute_init_model(self):
        if self.config.distribute_init_parameters:
            if self.__init_global_model_path is not None:
                with open(os.path.join(self.__init_global_model_path), "rb") as f:
                    self.cache_parameter_dict(pickle.load(f))
            else:
                self.cache_parameter_dict(self.tester.model_util.get_parameter_dict())
                # save GPU memory
                self.tester.offload_from_gpu()
            self.send_result(data={"parameter": self.cached_parameter_dict})

    def start(self):
        self._distribute_init_model()
        super().start()

    def _process_worker_data(self, worker_id, data):
        assert 0 <= worker_id < self.worker_number
        if data is not None:
            data = tensor_to(data, device=get_cpu_device())
            os.makedirs(os.path.join(self.save_dir, "worker_data"), exist_ok=True)
            data = DataStorage(
                data=data,
                data_path=os.path.join(self.save_dir, "worker_data", str(worker_id)),
            )
        self.__worker_data[worker_id] = data
        if len(self.__worker_data) == self.worker_number:
            self._round_number += 1
            self.__worker_data = {
                k: v for k, v in self.__worker_data.items() if v is not None
            }
            result = self._aggregate_worker_data(self.__worker_data)
            parameter = None
            if "parameter" in result:
                parameter = result["parameter"]
            if self._send_parameter_diff:
                if "parameter" in result:
                    get_logger().warning("send parameter diff")
                    result.pop("parameter")
                    result["parameter_diff"] = self.get_parameter_diff(parameter)
            self.send_result(result)
            if parameter is not None:
                self.cache_parameter_dict(parameter)
            self.__worker_data.clear()
            self.tester.offload_from_gpu()
        else:
            get_logger().debug(
                "we have %s committed, and we need %s workers,skip",
                len(self.__worker_data),
                self.worker_number,
            )

    def _aggregate_worker_data(self, worker_data):
        return self.__algorithm.aggregate_worker_data(
            worker_data=worker_data, old_parameter_dict=self.cached_parameter_dict
        )

    def _stopped(self) -> bool:
        return self._round_number >= self.config.round
