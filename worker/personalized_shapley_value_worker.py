from algorithm.fed_avg_algorithm import FedAVGAlgorithm

from .aggregation_worker import AggregationWorker
from .peer_worker import PeerWorker


class PersonalizedShapleyValueWorker(PeerWorker, AggregationWorker, FedAVGAlgorithm):
    def __init__(self, reuse_learning_rate=False, **kwargs):
        super().__init__(**kwargs)
        self.__reuse_learning_rate = reuse_learning_rate
        self._register_aggregation()

    def _aggretation(self, trainer, parameter_data):
        self.broadcast(
            (
                trainer.dataset_size,
                parameter_data,
            )
        )
        result = self.gather()
        result[self.worker_id] = (
            trainer.dataset_size,
            parameter_data,
        )
        parameter_data = self._aggregate_worker_data(
            round_number=self._round_num, worker_data=result
        )
        self._load_parameters(parameter_data, self.__reuse_learning_rate)
