from ..common_import import (AggregationServer, CentralizedAlgorithmFactory,
                             GraphAlgorithm, GraphWorker)
from .worker import FedGNNWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_gnn",
    client_cls=FedGNNWorker,
    server_cls=AggregationServer,
    algorithm_cls=GraphAlgorithm,
)
