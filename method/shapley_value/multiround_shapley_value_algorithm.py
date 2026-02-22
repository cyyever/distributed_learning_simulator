from cyy_torch_algorithm.shapley_value.multiround_shapley_value import (
    MultiRoundShapleyValue,
)

from algorithm.shapley_value_algorithm import ShapleyValueAlgorithm


class MultiRoundShapleyValueAlgorithm(ShapleyValueAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(MultiRoundShapleyValue, *args, **kwargs)
