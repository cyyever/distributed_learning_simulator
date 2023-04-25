from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import MachineLearningPhase

from .aggregation_worker import AggregationWorker


class FedAVGWorker(AggregationWorker):
    def _before_training(self):
        self.trainer.dataset_collection.remove_dataset(phase=MachineLearningPhase.Test)
        # load initial parameters
        if self.config.distribute_init_parameters:
            self.__get_result_from_server()
            if self._stopped():
                return
        self._register_aggregation()
        super()._before_training()

    def _offload_from_memory(self) -> None:
        super()._offload_from_memory()
        if self.config.offload_memory:
            self._model_cache.save()

    def __get_result_from_server(self) -> bool:
        self._offload_from_memory()
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
        self._offload_from_memory()
        self.__get_result_from_server()
