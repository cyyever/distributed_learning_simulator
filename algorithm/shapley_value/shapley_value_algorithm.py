import copy
import functools
import json
import os
from typing import Type

from algorithm.fed_avg_algorithm import FedAVGAlgorithm
from cyy_naive_lib.log import get_logger
from server.server import Server


class ShapleyValueAlgorithm(FedAVGAlgorithm):
    def __init__(self, sv_algorithm_cls: Type, server: Server, **kwargs):
        super().__init__(**kwargs)
        self._server: Server = server
        self.accumulate = False
        self.metric_type: str = "acc"
        self.sv_algorithm = None
        self.sv_algorithm_cls = sv_algorithm_cls
        self.choose_best_subset: bool = False
        self._server.need_init_performance = True

    def aggregate_worker_data(self) -> dict:
        if self.sv_algorithm is None:
            assert self._server.round_number == 1
            self.sv_algorithm = self.sv_algorithm_cls(
                worker_number=len(self._all_worker_data),
                last_round_metric=self._server.performance_stat[
                    self._server.round_number - 1
                ][f"test_{self.metric_type}"],
            )
            self.choose_best_subset = self._server.config.algorithm_kwargs.get(
                "choose_best_subset", False
            )
        self.sv_algorithm.set_metric_function(
            functools.partial(
                self._get_subset_metric, worker_data=copy.copy(self._all_worker_data)
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

    def _get_subset_metric(self, subset, worker_data) -> dict:
        assert subset
        worker_data = FedAVGAlgorithm._aggregate_worker_data(
            {k: v for k, v in worker_data.items() if k in subset}
        )

        assert worker_data
        return self._server.get_metric(worker_data, keep_performance_logger=False)[
            self.metric_type
        ]

    def exit(self) -> None:
        with open(
            os.path.join(self._server.config.save_dir, "shapley_values.json"),
            "wt",
            encoding="utf8",
        ) as f:
            json.dump(self.sv_algorithm.shapley_values, f)
        with open(
            os.path.join(self._server.config.save_dir, "shapley_values_S.json"),
            "wt",
            encoding="utf8",
        ) as f:
            json.dump(self.sv_algorithm.shapley_values_S, f)
