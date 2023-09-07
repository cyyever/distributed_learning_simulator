from server.fed_avg_server import FedAVGServer

from .multiround_shapley_value_algorithm import MultiRoundShapleyValueAlgorithm


class MultiRoundShapleyValueServer(FedAVGServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            **kwargs, algorithm=MultiRoundShapleyValueAlgorithm(server=self)
        )
