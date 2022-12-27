""" FedDropoutAvg: Generalizable federated learning for histopathology image classification (https://arxiv.org/pdf/2111.13230.pdf) """

from server.fed_avg_server import FedAVGServer

from .algorithm import FedDropoutAvgAlgorithm


class FedDropoutAvgServer(FedAVGServer):
    def __init__(self, **kwargs):
        super().__init__(algorithm=FedDropoutAvgAlgorithm(), **kwargs)
