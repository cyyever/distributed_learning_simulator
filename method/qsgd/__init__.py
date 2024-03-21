"""QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding  https://arxiv.org/abs/1610.02132"""

from ..common_import import (AggregationServer, CentralizedAlgorithmFactory,
                             FedAVGAlgorithm, GradientWorker,
                             StochasticQuantClientEndpoint,
                             StochasticQuantServerEndpoint)

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="QSGD",
    client_cls=GradientWorker,
    server_cls=AggregationServer,
    algorithm_cls=FedAVGAlgorithm,
    client_endpoint_cls=StochasticQuantClientEndpoint,
    server_endpoint_cls=StochasticQuantServerEndpoint,
)
