from distributed_learning_simulation import (
    AggregationServer,
    CentralizedAlgorithmFactory,
    FedAVGAlgorithm,
)

from .worker import QATWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_avg_qat",
    client_cls=QATWorker,
    server_cls=AggregationServer,
    algorithm_cls=FedAVGAlgorithm,
)
