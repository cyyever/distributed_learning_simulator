"""signSGD: Compressed Optimisation for Non-Convex Problems https://arxiv.org/abs/1802.04434"""

from distributed_learning_simulation import (
    AggregationServer,
    AlgorithmRepository,
)

from .algorithm import SignSGDAlgorithm
from .worker import SignSGDWorker

AlgorithmRepository.register_algorithm(
    algorithm_name="sign_SGD",
    client_cls=SignSGDWorker,
    server_cls=AggregationServer,
    algorithm_cls=SignSGDAlgorithm,
)
