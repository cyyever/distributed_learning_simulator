from cyy_naive_lib.log import get_logger

from .aggregation_worker import AggregationWorker
from .client import Client


class FedAVGWorker(Client, AggregationWorker):
    def __init__(self, reuse_learning_rate=False, **kwargs):
        super().__init__(**kwargs)
        self.__reuse_learning_rate = reuse_learning_rate
        self._register_aggregation()

    def before_training(self):
        # load initial parameters
        if not self.config.no_distribute_init_parameters:
            get_logger().debug("get init model")
            self.get_result_from_server()
            get_logger().debug("end get init model")

    def _aggretation(self, trainer, parameter_data):
        self.send_data_to_server(
            (
                len(trainer.dataset),
                parameter_data,
            )
        )
        self.get_result_from_server()

    def get_result_from_server(self):
        result = super().get_result_from_server()
        if result is not None:
            self._load_parameters(
                result, reuse_learning_rate=self.__reuse_learning_rate
            )
