from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import (MachineLearningPhase,
                                       StopExecutingException)

from .aggregation_worker import AggregationWorker
from .client import Client


class FedAVGWorker(Client, AggregationWorker):
    def __init__(self, **kwargs):
        Client.__init__(self, **kwargs)
        AggregationWorker.__init__(self, self.trainer)

    def _before_training(self):
        self.trainer.dataset_collection.remove_dataset(phase=MachineLearningPhase.Test)
        # load initial parameters
        if self.config.distribute_init_parameters:
            self.__get_result_from_server()
            if self._stopped():
                return
        self._register_aggregation()
        super()._before_training()

    def _offload_from_memory(self):
        self._model_cache.save(self.save_dir)
        super()._offload_from_memory()

    def __get_result_from_server(self) -> bool:
        while True:
            result = super()._get_result_from_server()
            get_logger().debug("get result from server %s", type(result))
            if result is None:
                get_logger().debug("skip round %s", self._round_num)
                self._round_num += 1
                self.send_data_to_server(None)
                if self._stopped():
                    break
                continue
            self._load_result_from_server(result=result)
            return True
        return False

    def _aggregation(self, sent_data):
        self.send_data_to_server(sent_data)
        self.__get_result_from_server()

    def _load_result_from_server(self, result):
        if "end_training" in result:
            self._force_stop = True
            raise StopExecutingException()
        if "parameter_diff" in result:
            parameter = {}
            for k, v in result["parameter_diff"].items():
                parameter[k] = self._model_cache.cached_parameter_dict[k] + v
            result["parameter"] = parameter
        self._load_parameters(result["parameter"])
