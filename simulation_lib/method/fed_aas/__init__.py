"""
Our work
"""
from ..algorithm_factory import CentralizedAlgorithmFactory
from ..common_import import (DifferentialPrivacyEmbeddingEndpoint,
                             GraphNodeEmbeddingPassingAlgorithm)
from .server import FedAASServer
from .worker import FedAASWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_aas",
    client_cls=FedAASWorker,
    server_cls=FedAASServer,
    algorithm_cls=GraphNodeEmbeddingPassingAlgorithm,
)
CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_aas_dp",
    client_cls=FedAASWorker,
    server_cls=FedAASServer,
    algorithm_cls=GraphNodeEmbeddingPassingAlgorithm,
    client_endpoint_cls=DifferentialPrivacyEmbeddingEndpoint,
)
