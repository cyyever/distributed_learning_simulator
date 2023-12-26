from ..algorithm_factory import CentralizedAlgorithmFactory
from ..common_import import (NNADQClientEndpoint, NNADQServerEndpoint,
                             StochasticQuantClientEndpoint,
                             StochasticQuantServerEndpoint)
from .server import FedOBDServer
from .worker import FedOBDWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_obd",
    client_cls=FedOBDWorker,
    server_cls=FedOBDServer,
    client_endpoint_cls=NNADQClientEndpoint,
    server_endpoint_cls=NNADQServerEndpoint,
)

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_obd_sq",
    client_cls=FedOBDWorker,
    server_cls=FedOBDServer,
    client_endpoint_cls=StochasticQuantClientEndpoint,
    server_endpoint_cls=StochasticQuantServerEndpoint,
)
