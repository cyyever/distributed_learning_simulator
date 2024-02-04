from ..algorithm_factory import CentralizedAlgorithmFactory
from ..common_import import (AggregationServer,
                             GraphNodeEmbeddingPassingAlgorithm,
                             GraphWorker)

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_gnn",
    client_cls=GraphWorker,
    server_cls=AggregationServer,
    algorithm_cls=GraphNodeEmbeddingPassingAlgorithm,
)
