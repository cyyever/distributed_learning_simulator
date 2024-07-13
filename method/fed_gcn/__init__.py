"""
FedGCN: Convergence and Communication Tradeoffs in Federated Training of Graph Convolutional Networks
"""

from distributed_learning_simulation import (AggregationServer,
                                             CentralizedAlgorithmFactory,
                                             GraphAlgorithm)

from .worker import FedGCNWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_gcn",
    client_cls=FedGCNWorker,
    server_cls=AggregationServer,
    algorithm_cls=GraphAlgorithm,
)
