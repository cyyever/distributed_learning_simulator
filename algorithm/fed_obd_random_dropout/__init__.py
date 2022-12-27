from _algorithm_factory import CentralizedAlgorithmFactory
from algorithm.fed_obd.server import FedOBDServer
from topology.quantized_endpoint import (NNADQClientEndpoint,
                                         NNADQServerEndpoint)

from .worker import FedOBDRandomDropoutWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_obd_random_dropout",
    client_cls=FedOBDRandomDropoutWorker,
    server_cls=FedOBDServer,
    client_endpoint_cls=NNADQClientEndpoint,
    server_endpoint_cls=NNADQServerEndpoint,
)
