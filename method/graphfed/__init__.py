from ..common_import import CentralizedAlgorithmFactory
from .algorithm import GraphFedAlgorithm
from .server import GraphFedServer
from .worker import GraphFedWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="graphfed",
    client_cls=GraphFedWorker,
    server_cls=GraphFedServer,
    algorithm_cls=GraphFedAlgorithm,
)
