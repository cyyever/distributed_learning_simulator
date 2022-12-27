from _algorithm_factory import CentralizedAlgorithmFactory

from .server import FedDropoutAvgServer
from .worker import FedDropoutAvgWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_dropout_avg",
    client_cls=FedDropoutAvgWorker,
    server_cls=FedDropoutAvgServer,
)
