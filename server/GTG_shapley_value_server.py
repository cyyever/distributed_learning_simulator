import functools

from cyy_torch_algorithm.shapely_value.gtg_shapley_value import GTGShapleyValue

from .fed_avg_server import FedAVGServer
from .shapley_value_algorithm import ShapleyValueAlgorithm


class GTGShapleyValueServer(FedAVGServer, ShapleyValueAlgorithm):

    def _aggregate_worker_data_impl(self, worker_data):
        if self.algorithm is None:
            self.algorithm = GTGShapleyValue(
                worker_number=self.worker_number,
                last_round_metric=self.get_metric(self.cached_parameter_dict)[
                    self.metric_type
                ],
            )
            self.algorithm.set_metric_function(
                functools.partial(self._get_subset_metric, worker_data=worker_data)
            )
        self.algorithm.compute()
        return super()._aggregate_worker_data_impl(worker_data)
