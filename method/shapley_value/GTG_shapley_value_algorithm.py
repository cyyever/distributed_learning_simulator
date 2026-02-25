from algorithm.shapley_value_algorithm import ShapleyValueAlgorithm
from cyy_torch_algorithm.shapley_value.gtg_shapley_value import GTGShapleyValue


class GTGShapleyValueAlgorithm(ShapleyValueAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(GTGShapleyValue, *args, **kwargs)
