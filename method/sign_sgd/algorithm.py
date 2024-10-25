"""signSGD: Compressed Optimisation for Non-Convex Problems https://arxiv.org/abs/1802.04434"""

from distributed_learning_simulation import FedAVGAlgorithm, ParameterMessage


class SignSGDAlgorithm(FedAVGAlgorithm):
    def aggregate_worker_data(self) -> ParameterMessage:
        message = super().aggregate_worker_data()
        assert isinstance(message, ParameterMessage)
        message.parameter = {k: v.sign() for k, v in message.parameter.items()}
        return message
