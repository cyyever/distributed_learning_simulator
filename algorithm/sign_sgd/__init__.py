from _algorithm_factory import CentralizedAlgorithmFactory

from .server import SignSGDServer
from .worker import SignSGDWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="sign_SGD", client_cls=SignSGDWorker, server_cls=SignSGDServer
)
