from ..algorithm_factory import CentralizedAlgorithmFactory
from ..common_import import GraphNodeServer, GraphWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_gnn",
    client_cls=GraphWorker,
    server_cls=GraphNodeServer,
)
