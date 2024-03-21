from .multiround_shapley_value_algorithm import MultiRoundShapleyValueAlgorithm
from .shapley_value_server import ShapleyValueServer


class MultiRoundShapleyValueServer(ShapleyValueServer):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            **kwargs, algorithm=MultiRoundShapleyValueAlgorithm(server=self)
        )
