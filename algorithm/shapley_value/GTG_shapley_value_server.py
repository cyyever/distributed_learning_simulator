import functools

from algorithm.fed_avg_algorithm import FedAVGAlgorithm
from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.shapely_value.gtg_shapley_value import GTGShapleyValue
from server.fed_avg_server import FedAVGServer


class GTGShapleyValueServer(FedAVGServer):
    def __init__(self, *args, **kwargs):
        algorithm = FedAVGAlgorithm()
        algorithm.accumulate = False
        super().__init__(*args, **kwargs, algorithm=algorithm)
        self.metric_type: str = "acc"
        self.sv_algorithm = None
        self.choose_best_subset: bool = self.config.algorithm_kwargs.get(
            "choose_best_subset", False
        )

    def _aggregate_worker_data(self, worker_data):
        if self.sv_algorithm is None:
            self.sv_algorithm = GTGShapleyValue(
                worker_number=self.worker_number,
                last_round_metric=self.get_metric(self.get_parameter_dict)[
                    self.metric_type
                ],
            )
        self.sv_algorithm.set_metric_function(
            functools.partial(self._get_subset_metric, worker_data=worker_data)
        )
        self.sv_algorithm.compute()
        if self.choose_best_subset:
            best_subset: set = set(
                self.sv_algorithm.shapley_values_S[
                    self.sv_algorithm.round_number
                ].keys()
            )
            if best_subset:
                get_logger().warning("use subset %s", best_subset)
                worker_data = {k: v for k, v in worker_data.items() if k in best_subset}
        return super()._aggregate_worker_data(worker_data)

    def _get_subset_metric(self, _, subset, worker_data):
        worker_data = {k: v for k, v in worker_data.items() if k in subset}
        assert worker_data
        return self.get_metric(super()._aggregate_worker_data(worker_data))[
            self.metric_type
        ]
