import functools

from algorithm.fed_avg_algorithm import FedAVGAlgorithm
from cyy_naive_lib.log import get_logger


class ShapleyValueAlgorithm(FedAVGAlgorithm):
    def __init__(self, sv_algorithm_cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulate = False
        self.metric_type: str = "acc"
        self.sv_algorithm = None
        self.sv_algorithm_cls = sv_algorithm_cls
        self.choose_best_subset: bool = False

    def aggregate_worker_data(self):
        if self.sv_algorithm is None:
            self.sv_algorithm = self.sv_algorithm_cls(
                worker_number=len(self._all_worker_data),
                last_round_metric=self._server.get_metric(
                    self._server._model_cache.get_parameter_dict
                )[self.metric_type],
            )
            self.choose_best_subset = self._server.config.algorithm_kwargs.get(
                "choose_best_subset", False
            )
        self.sv_algorithm.set_metric_function(
            functools.partial(
                self._get_subset_metric, worker_data=self._all_worker_data
            )
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
                self._all_worker_data = {
                    k: v for k, v in self._all_worker_data.items() if k in best_subset
                }
        return super().aggregate_worker_data()

    def _get_subset_metric(self, _, subset, worker_data):
        worker_data = {k: v for k, v in worker_data.items() if k in subset}
        assert worker_data
        worker_data_backup = self._all_worker_data
        self._all_worker_data = worker_data
        metric = self._server.get_metric(super().aggregate_worker_data())[
            self.metric_type
        ]
        self._all_worker_data = worker_data_backup
        return metric
