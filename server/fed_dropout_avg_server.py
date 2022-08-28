""" FedDropoutAvg: Generalizable federated learning for histopathology image classification (https://arxiv.org/pdf/2111.13230.pdf) """
from algorithm.fed_dropout_avg_algorithm import FedDropoutAvgAlgorithm

from .fed_avg_random_subset_server import FedAVGRandomSubsetServer


class FedDropoutAvgServer(FedAVGRandomSubsetServer):
    def __init__(self, **kwargs):
        super().__init__(algorithm=FedDropoutAvgAlgorithm(), **kwargs)
