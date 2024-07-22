import os
import sys

from cyy_torch_algorithm.shapely_value.multiround_shapley_value import \
    MultiRoundShapleyValue

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from algorithm.shapley_value_algorithm import ShapleyValueAlgorithm


class MultiRoundShapleyValueAlgorithm(ShapleyValueAlgorithm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(MultiRoundShapleyValue, *args, **kwargs)
