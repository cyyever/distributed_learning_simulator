"""
Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
"""

from distributed_learning_simulation import (AggregationServer,
                                             CentralizedAlgorithmFactory)

from .algorithm import FedClusterGCNAlgorithm
from .worker import FedClusterGCNWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_cluster_gcn",
    client_cls=FedClusterGCNWorker,
    server_cls=AggregationServer,
    algorithm_cls=FedClusterGCNAlgorithm,
)
