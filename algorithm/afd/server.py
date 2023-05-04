""" Adaptive Federated Dropout: Improving Communication Efficiency and Generalization for Federated Learning (https://arxiv.org/abs/2011.04050)"""


from server.fed_avg_server import FedAVGServer

from .algorithm import SingleModelAdaptiveFedDropoutAlgorithm


class SingleModelAdaptiveFedDropoutServer(FedAVGServer):
    def __init__(self, config, **kwargs):
        super().__init__(
            **kwargs,
            config=config,
            algorithm=SingleModelAdaptiveFedDropoutAlgorithm(config=config)
        )
