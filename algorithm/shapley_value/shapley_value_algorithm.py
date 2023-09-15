import json
import os
from typing import Iterable, Type

from algorithm.fed_avg_algorithm import FedAVGAlgorithm
from cyy_naive_lib.log import get_logger
from server.fed_avg_server import FedAVGServer


class ShapleyValueAlgorithm(FedAVGAlgorithm):
    def __init__(self, sv_algorithm_cls: Type, server: FedAVGServer, **kwargs):
        super().__init__(**kwargs)
        self._server: FedAVGServer = server
        self.accumulate = False
        self.metric_type: str = "acc"
        self.sv_algorithm = None
        self.sv_algorithm_cls = sv_algorithm_cls
        self.shapley_values: dict = {}
        self.shapley_values_S: dict = {}
        setattr(self._server, "need_init_performance", True)

    @property
    def config(self):
        return self._server.config

    @property
    def algorithm_kwargs(self):
        return self._server.config.algorithm_kwargs

    @property
    def choose_best_subset(self) -> bool:
        return self.algorithm_kwargs.get("choose_best_subset", False)

    def _get_players(self) -> Iterable:
        return sorted(self._all_worker_data.keys())

    def aggregate_worker_data(self) -> dict:
        if self.sv_algorithm is None:
            assert self._server.round_number == 1
            self.sv_algorithm = self.sv_algorithm_cls(
                players=self._get_players(),
                last_round_metric=self._server.performance_stat[
                    self._server.round_number - 1
                ][f"test_{self.metric_type}"],
            )
        assert self.sv_algorithm is not None
        self.sv_algorithm.set_metric_function(self._get_subset_metric)
        self.sv_algorithm.compute(round_number=self._server.round_number)
        self.shapley_values[self._server.round_number] = self._convert_shapley_values(
            self.sv_algorithm.shapley_values
        )
        self.shapley_values_S[self._server.round_number] = self._convert_shapley_values(
            self.sv_algorithm.shapley_values_S
        )
        if self.choose_best_subset:
            best_subset: set = set(
                self.shapley_values_S[self._server.round_number].keys()
            )
            if best_subset:
                get_logger().warning("use subset %s", best_subset)
                self._all_worker_data = {
                    k: v for k, v in self._all_worker_data.items() if k in best_subset
                }
        return super().aggregate_worker_data()

    def _convert_shapley_values(self, shapley_values: dict) -> dict:
        return shapley_values

    def _get_subset_metric(self, subset) -> dict:
        assert subset
        worker_data = FedAVGAlgorithm._aggregate_worker_data(
            {k: v for k, v in self._all_worker_data.items() if k in subset}
        )

        assert worker_data
        return self._server.get_metric(worker_data, keep_performance_logger=False)[
            self.metric_type
        ]

    def exit(self) -> None:
        assert self.sv_algorithm is not None
        with open(
            os.path.join(self._server.config.save_dir, "shapley_values.json"),
            "wt",
            encoding="utf8",
        ) as f:
            json.dump(self.shapley_values, f)
        if self.choose_best_subset:
            with open(
                os.path.join(self._server.config.save_dir, "shapley_values_S.json"),
                "wt",
                encoding="utf8",
            ) as f:
                json.dump(self.shapley_values_S, f)
