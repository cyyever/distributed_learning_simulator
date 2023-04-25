from _algorithm_factory import CentralizedAlgorithmFactory
from algorithm.fed_obd.server import FedOBDServer
from topology.quantized_endpoint import (NNADQClientEndpoint,
                                         NNADQServerEndpoint)

from .server import FedOBDLayerServer
from .worker import FedOBDLayerWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_obd_layer",
    client_cls=FedOBDLayerWorker,
    server_cls=FedOBDLayerServer,
    client_endpoint_cls=NNADQClientEndpoint,
    server_endpoint_cls=NNADQServerEndpoint,
)
