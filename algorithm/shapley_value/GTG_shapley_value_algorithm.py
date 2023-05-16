from cyy_torch_algorithm.shapely_value.gtg_shapley_value import GTGShapleyValue

from .shapley_value_algorithm import ShapleyValueAlgorithm


class GTGShapleyValueAlgorithm(ShapleyValueAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(GTGShapleyValue, *args, **kwargs)
