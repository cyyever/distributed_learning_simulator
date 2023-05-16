from algorithm.fed_avg_algorithm import FedAVGAlgorithm
from cyy_torch_algorithm.shapely_value.multiround_shapley_value import \
    MultiRoundShapleyValue


class MultiRoundShapleyValueAlgorithm(FedAVGAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(MultiRoundShapleyValue, *args, **kwargs)
