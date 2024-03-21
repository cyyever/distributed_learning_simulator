from .GTG_shapley_value_algorithm import GTGShapleyValueAlgorithm
from .shapley_value_server import ShapleyValueServer


class GTGShapleyValueServer(ShapleyValueServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, algorithm=GTGShapleyValueAlgorithm(server=self))
