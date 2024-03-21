from ..common_import import (AggregationServer, AggregationWorker,
                             CentralizedAlgorithmFactory, FedAVGAlgorithm)

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_avg",
    client_cls=AggregationWorker,
    server_cls=AggregationServer,
    algorithm_cls=FedAVGAlgorithm,
)
