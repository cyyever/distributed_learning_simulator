from _algorithm_factory import CentralizedAlgorithmFactory

from server.fed_avg_server import FedAVGServer
from worker.fed_avg_worker import FedAVGWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_avg",
    client_cls=FedAVGWorker,
    server_cls=FedAVGServer,
)
