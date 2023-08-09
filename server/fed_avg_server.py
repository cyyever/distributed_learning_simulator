import json
import os
import pickle

from algorithm.fed_avg_algorithm import FedAVGAlgorithm
from cyy_naive_lib.log import get_logger

from .aggregation_server import AggregationServer


class FedAVGServer(AggregationServer):
    def __init__(self, *args, **kwargs):
        if "algorithm" not in kwargs:
            kwargs["algorithm"] = FedAVGAlgorithm()
        super().__init__(*args, **kwargs)
        self._compute_stat: bool = True
        self.__stat: dict = {}
        self.__plateau = 0
        self.__max_acc = 0
        self._early_stop = self.config.algorithm_kwargs.get("early_stop", False)
        if self._early_stop:
            get_logger().warning("stop early")

    def _after_aggregate_worker_data(self, result):
        if "parameter" in result:
            if self._compute_stat or self._early_stop:
                self._record_compute_stat(result["parameter"])
            if self._early_stop and self._convergent():
                result["end_training"] = True
        super()._after_aggregate_worker_data(result=result)

    @property
    def performance_stat(self) -> dict:
        return self.__stat

    def _record_compute_stat(self, parameter_dict: dict) -> None:
        self.tester.set_visualizer_prefix(f"round: {self._round_number},")
        metric = self.get_metric(parameter_dict)

        round_stat = {}
        round_stat["test_loss"] = metric["loss"]
        round_stat["test_acc"] = metric["acc"]

        self.__stat[self.round_number] = round_stat
        os.makedirs(self.config.save_dir, exist_ok=True)
        with open(
            os.path.join(self.config.save_dir, "round_record.json"),
            "wt",
            encoding="utf8",
        ) as f:
            json.dump(self.__stat, f)

        max_acc = max(t["test_acc"] for t in self.__stat.values())
        if max_acc > self.__max_acc:
            self.__max_acc = max_acc
            with open(
                os.path.join(self.config.save_dir, "best_global_model.pk"), "wb"
            ) as f:
                pickle.dump(
                    parameter_dict,
                    f,
                )

    def _convergent(self) -> bool:
        max_acc = max(t["test_acc"] for t in self.__stat.values())
        diff = 0.001
        if max_acc > self.__max_acc + diff:
            self.__max_acc = max_acc
            self.__plateau = 0
            return False
        del max_acc
        get_logger().error(
            "max acc is %s diff is %s",
            self.__max_acc,
            self.__max_acc - self.__stat[self.round_number]["test_acc"],
        )
        self.__plateau += 1
        get_logger().error("plateau is %s", self.__plateau)
        if self.__plateau >= 5:
            return True
        return False
