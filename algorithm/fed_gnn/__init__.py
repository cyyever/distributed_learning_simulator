from _algorithm_factory import CentralizedAlgorithmFactory
from server.graph_server import GraphNodeServer

from worker.graph_worker import GraphWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_gnn",
    client_cls=GraphWorker,
    server_cls=GraphNodeServer,
)
