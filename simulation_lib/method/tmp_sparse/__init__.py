# https://www.overleaf.com/project/650262e57b008ad1f883ca66
from ..algorithm_factory import CentralizedAlgorithmFactory
from ..common_import import (AggregationServer, FedAVGAlgorithm)
from .worker import FedSparseWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_sparse",
    client_cls=FedSparseWorker,
    server_cls=AggregationServer,
    algorithm_cls=FedAVGAlgorithm,
)
