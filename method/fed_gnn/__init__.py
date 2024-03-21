from ..common_import import (AggregationServer, CentralizedAlgorithmFactory,
                             GraphAlgorithm, GraphWorker)

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_gnn",
    client_cls=GraphWorker,
    server_cls=AggregationServer,
    algorithm_cls=GraphAlgorithm,
)
