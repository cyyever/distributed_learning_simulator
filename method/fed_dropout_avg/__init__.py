"""FedDropoutAvg: Generalizable federated learning for histopathology image classification (https://arxiv.org/pdf/2111.13230.pdf)"""

from distributed_learning_simulation import (
    AggregationServer,
    CentralizedAlgorithmFactory,
)

from .algorithm import FedDropoutAvgAlgorithm
from .worker import FedDropoutAvgWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_dropout_avg",
    client_cls=FedDropoutAvgWorker,
    server_cls=AggregationServer,
    algorithm_cls=FedDropoutAvgAlgorithm,
)
