"""
Our work
"""
from ..algorithm_factory import CentralizedAlgorithmFactory
from ..fed_aas.server import FedAASServer
from .worker import FedCSSWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_css", client_cls=FedCSSWorker, server_cls=FedAASServer
)
