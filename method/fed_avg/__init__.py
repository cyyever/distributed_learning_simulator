from distributed_learning_simulation import (
    AggregationServer,
    AggregationWorker,
    AlgorithmRepository,
    FedAVGAlgorithm,
)

AlgorithmRepository.register_algorithm(
    algorithm_name="fed_avg",
    client_cls=AggregationWorker,
    server_cls=AggregationServer,
    algorithm_cls=FedAVGAlgorithm,
)
