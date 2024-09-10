import json
import os
from typing import Any, Type

from cyy_naive_lib.concurrency import batch_process
from cyy_naive_lib.log import log_error, log_warning
from cyy_torch_algorithm.shapely_value.shapley_value import \
    RoundBasedShapleyValue
from cyy_torch_toolbox import TorchProcessTaskQueue
from distributed_learning_simulation import (AggregationServer,
                                             FedAVGAlgorithm, ParameterMessage)


class ShapleyValueAlgorithm(FedAVGAlgorithm):
    def __init__(
        self, sv_algorithm_cls: Type, server: AggregationServer, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._server: AggregationServer = server
        self.accumulate = False
        self.metric_type: str = "accuracy"
        self.__sv_algorithm: None | RoundBasedShapleyValue = None
        self.sv_algorithm_cls = sv_algorithm_cls

    @property
    def sv_algorithm(self) -> RoundBasedShapleyValue:
        if self.__sv_algorithm is None:
            assert self._all_worker_data
            assert self._server.round_index == 1
            self.__sv_algorithm = self.sv_algorithm_cls(
                players=sorted(self._all_worker_data.keys()),
                initial_metric=self._server.performance_stat[
                    self._server.round_index - 1
                ][f"test_{self.metric_type}"],
                algorithm_kwargs=self.config.algorithm_kwargs,
            )
            assert isinstance(self.__sv_algorithm, RoundBasedShapleyValue)
            if (
                self.config.algorithm_kwargs.get("round_trunc_threshold", None)
                is not None
            ):
                self.__sv_algorithm.set_round_truncation_threshold(
                    self.config.algorithm_kwargs["round_trunc_threshold"]
                )
            self.sv_algorithm.set_batch_metric_function(self._get_batch_metric)
        # For client selection in each round
        self.__sv_algorithm.set_players(
            sorted({k for k, v in self._all_worker_data.items() if v is not None})
        )
        return self.__sv_algorithm

    @property
    def choose_best_subset(self) -> bool:
        return self.config.algorithm_kwargs.get("choose_best_subset", False)

    def aggregate_worker_data(self) -> ParameterMessage:
        self.sv_algorithm.compute(round_index=self._server.round_index)
        if self.choose_best_subset:
            assert hasattr(self.sv_algorithm, "shapley_values_S")
            best_players = self.sv_algorithm.get_best_players(
                round_index=self._server.round_index
            )
            assert best_players is not None
            log_warning("use players %s", best_players)
            self._all_worker_data = {k: self._all_worker_data[k] for k in best_players}
        return super().aggregate_worker_data()

    def _batch_metric_worker(self, task, **kwargs) -> dict:
        return {task: self._get_subset_metric(subset=task)}

    def _get_batch_metric(self, subsets) -> dict:
        if len(subsets) == 1:
            return {list(subsets)[0]: self._get_subset_metric(list(subsets)[0])}
        queue = TorchProcessTaskQueue(
            worker_num=self.config.algorithm_kwargs.get("sv_worker_number", None)
        )
        queue.disable_logger()
        queue.start(worker_fun=self._batch_metric_worker)
        res = batch_process(queue, subsets)
        queue.stop()
        return res

    def _get_subset_metric(self, subset) -> float:
        assert subset
        aggregated_parameter = FedAVGAlgorithm.aggregate_parameter(
            {k: self._all_worker_data[k] for k in self.sv_algorithm.get_players(subset)}
        )
        assert aggregated_parameter
        return self._server.get_metric(
            aggregated_parameter, log_performance_metric=False
        )[self.metric_type]

    def exit(self) -> None:
        assert self.sv_algorithm is not None
        self.sv_algorithm.exit()
        with open(
            os.path.join(self.config.save_dir, "shapley_values.json"),
            "wt",
            encoding="utf8",
        ) as f:
            json.dump(self.sv_algorithm.get_result(), f)
