import json
import os
from typing import Any, Type

from cyy_naive_lib.concurrency import batch_process
from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.shapely_value.shapley_value import \
    RoundBasedShapleyValue
from cyy_torch_toolbox.concurrency import TorchProcessTaskQueue
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
            )
            assert isinstance(self.__sv_algorithm, RoundBasedShapleyValue)
            if (
                self.config.algorithm_kwargs.get("round_trunc_threshold", None)
                is not None
            ):
                self.__sv_algorithm.set_round_truncation_threshold(
                    self.config.algorithm_kwargs["round_trunc_threshold"]
                )
            self.__sv_algorithm.config = self.config
            self.sv_algorithm.set_batch_metric_function(self._get_batch_metric)
        return self.__sv_algorithm

    @property
    def choose_best_subset(self) -> bool:
        return self.config.algorithm_kwargs.get("choose_best_subset", False)

    def aggregate_worker_data(self) -> ParameterMessage:
        self.sv_algorithm.compute(round_number=self._server.round_index)
        if self.choose_best_subset:
            assert hasattr(self.sv_algorithm, "shapley_values_S")
            best_subset = self.sv_algorithm.get_best_players(
                round_number=self._server.round_index
            )
            assert best_subset is not None
            get_logger().warning("use subset %s", best_subset)
            self._all_worker_data = {
                k: v for k, v in self._all_worker_data.items() if k in best_subset
            }
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

    def _get_subset_metric(self, subset) -> dict:
        assert subset
        aggregated_parameter = FedAVGAlgorithm.aggregate_parameter(
            {k: v for k, v in self._all_worker_data.items() if k in subset}
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
