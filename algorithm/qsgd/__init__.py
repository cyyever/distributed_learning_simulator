from _algorithm_factory import CentralizedAlgorithmFactory
from topology.quantized_endpoint import (StochasticQuantClientEndpoint,
                                         StochasticQuantServerEndpoint)
from worker.gradient_worker import GradientWorker

from .server import QSGDServer

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="QSGD",
    client_cls=GradientWorker,
    server_cls=QSGDServer,
    client_endpoint_cls=StochasticQuantClientEndpoint,
    server_endpoint_cls=StochasticQuantServerEndpoint,
)
