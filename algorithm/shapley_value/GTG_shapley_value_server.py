from server.fed_avg_server import FedAVGServer

from .GTG_shapley_value_algorithm import GTGShapleyValueAlgorithm


class GTGShapleyValueServer(FedAVGServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, algorithm=GTGShapleyValueAlgorithm(server=self))
