from algorithm.fed_avg_algorithm import FedAVGAlgorithm


class ShapleyValueAlgorithm(FedAVGAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_type = "acc"
        self.algorithm = None

    def _get_subset_metric(self, _, subset, worker_data):
        worker_data = {k: v for k, v in worker_data.items() if k in subset}
        assert worker_data
        return self.get_metric(self._aggregate_worker_data_impl(worker_data))[
            self.metric_type
        ]
