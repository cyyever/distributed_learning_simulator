""" signSGD: Compressed Optimisation for Non-Convex Problems https://arxiv.org/abs/1802.04434 """

from server.gradient_server import GradientServer

from .aggregation_algorithm import SignSGDAlgorithm


class SignSGDServer(GradientServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, algorithm=SignSGDAlgorithm(), **kwargs)
