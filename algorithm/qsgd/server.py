"""QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding  https://arxiv.org/abs/1610.02132"""

from server.gradient_server import GradientServer

from .aggregation_algorithm import QSGDAlgorithm


class QSGDServer(GradientServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, algorithm=QSGDAlgorithm(), **kwargs)
