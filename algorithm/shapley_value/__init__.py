from _algorithm_factory import CentralizedAlgorithmFactory
from worker.fed_avg_worker import FedAVGWorker

from .GTG_shapley_value_server import GTGShapleyValueServer
from .multiround_shapley_value_server import MultiRoundShapleyValueServer

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="multiround_shapley_value",
    client_cls=FedAVGWorker,
    server_cls=MultiRoundShapleyValueServer,
)
CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="GTG_shapley_value",
    client_cls=FedAVGWorker,
    server_cls=GTGShapleyValueServer,
)
