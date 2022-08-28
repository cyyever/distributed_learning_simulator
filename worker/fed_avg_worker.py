from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import (MachineLearningPhase,
                                       StopExecutingException)
from util.model_cache import ModelCache

from .aggregation_worker import AggregationWorker
from .client import Client


class FedAVGWorker(Client, AggregationWorker, ModelCache):
    _send_parameter_diff = True
    __stop = False

    def __init__(self, **kwargs):
        Client.__init__(self, **kwargs)
        AggregationWorker.__init__(self)
        ModelCache.__init__(self)

    def _before_training(self):
        super()._before_training()
        self.trainer.model_util.disable_running_stats()
        self.trainer.dataset_collection.remove_dataset(phase=MachineLearningPhase.Test)
        # load initial parameters
        if self.config.distribute_init_parameters:
            self.get_result_from_server()
        self._register_aggregation()

    def _load_to_memory(self):
        super()._load_to_memory()
        self._load_cached_model_to_memory(self.save_dir)

    def _offload_from_memory(self):
        self._offload_cached_model_from_memory(self.save_dir)
        super()._offload_from_memory()

    def _stopped(self):
        if super()._stopped():
            return True
        return self.__stop

    def get_result_from_server(self):
        while True:
            result = super().get_result_from_server()
            if result is None:
                get_logger().warning("skip round %s", self._round_num)
                self._round_num += 1
                self.send_data_to_server(None)
                if self._stopped():
                    break
                continue
            self.load_result_from_server(result=result)
            return

    def _aggretation(self, sent_data):
        if self._send_parameter_diff:
            parameter_data = sent_data.pop("parameter")
            parameter_diff = self.get_parameter_diff(parameter_data)
            sent_data["parameter_diff"] = parameter_diff
        self.send_data_to_server(sent_data)
        self.get_result_from_server()

    def _get_sent_data(self) -> dict:
        sent_data = super()._get_sent_data()
        sent_data |= {"dataset_size": self.trainer.dataset_size}
        return sent_data

    def load_result_from_server(self, result):
        if "end_training" in result:
            self.__stop = True
            raise StopExecutingException()
        self._load_parameters(result["parameter"])
        self.cache_parameter_dict(self.trainer.model_util.get_parameter_dict())
        self.trainer.model_util.disable_running_stats()
