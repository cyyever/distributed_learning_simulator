from distributed_learning_simulation import (
    AggregationServer,
    AlgorithmRepository,
    FedAVGAlgorithm,
)

from .worker import QATWorker

AlgorithmRepository.register_algorithm(
    algorithm_name="fed_avg_qat",
    client_cls=QATWorker,
    server_cls=AggregationServer,
    algorithm_cls=FedAVGAlgorithm,
)
