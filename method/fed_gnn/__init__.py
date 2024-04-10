from distributed_learning_simulation import (AggregationServer,
                                             CentralizedAlgorithmFactory,
                                             GraphAlgorithm)

from .worker import FedGNNWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_gnn",
    client_cls=FedGNNWorker,
    server_cls=AggregationServer,
    algorithm_cls=GraphAlgorithm,
)
