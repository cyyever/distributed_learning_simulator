# FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization (https://arxiv.org/abs/1909.13014)
from ..common_import import (AggregationServer, AggregationWorker,
                             CentralizedAlgorithmFactory, FedAVGAlgorithm,
                             StochasticQuantClientEndpoint,
                             StochasticQuantServerEndpoint)

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_paq",
    client_cls=AggregationWorker,
    server_cls=AggregationServer,
    client_endpoint_cls=StochasticQuantClientEndpoint,
    server_endpoint_cls=StochasticQuantServerEndpoint,
    algorithm_cls=FedAVGAlgorithm,
)
