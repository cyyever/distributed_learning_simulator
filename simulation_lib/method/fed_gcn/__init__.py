"""
FedGCN: Convergence and Communication Tradeoffs in Federated Training of Graph Convolutional Networks
"""
from ..algorithm_factory import CentralizedAlgorithmFactory
from ..common_import import (AggregationServer,
                             GraphNodeEmbeddingPassingAlgorithm)
from .worker import FedGCNWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_gcn",
    client_cls=FedGCNWorker,
    server_cls=AggregationServer,
    algorithm_cls=GraphNodeEmbeddingPassingAlgorithm,
)
