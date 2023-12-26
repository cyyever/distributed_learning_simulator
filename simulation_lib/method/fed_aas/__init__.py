"""
Our work
"""
from ..algorithm_factory import CentralizedAlgorithmFactory
from .server import FedAASServer
from .worker import FedAASWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_aas",
    client_cls=FedAASWorker,
    server_cls=FedAASServer,
)
