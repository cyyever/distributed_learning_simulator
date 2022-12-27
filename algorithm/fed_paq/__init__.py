# FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization (https://arxiv.org/abs/1909.13014)
from _algorithm_factory import CentralizedAlgorithmFactory
from topology.quantized_endpoint import (StochasticQuantClientEndpoint,
                                         StochasticQuantServerEndpoint)

from server.fed_avg_server import FedAVGServer
from worker.fed_avg_worker import FedAVGWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_paq",
    client_cls=FedAVGWorker,
    server_cls=FedAVGServer,
    client_endpoint_cls=StochasticQuantClientEndpoint,
    server_endpoint_cls=StochasticQuantServerEndpoint,
)
