import math
from typing import Any

from cyy_naive_lib.log import get_logger

from ..common_import import GraphNodeServer, ParameterMessageBase


class FedAASServer(GraphNodeServer):
    need_init_performance = True
    last_interval: int = 0

    def start(self) -> None:
        assert self.config.distribute_init_parameters
        assert self._compute_stat
        self.last_interval = self.config.algorithm_kwargs["batch_number"]
        super().start()

    def _before_send_result(self, result: Any) -> None:
        super()._before_send_result(result=result)
        if not isinstance(result, ParameterMessageBase):
            return
        fixed_sharing_interval = self.config.algorithm_kwargs.get(
            "fixed_sharing_interval", None
        )
        if fixed_sharing_interval is not None:
            result.other_data["sharing_interval"] = fixed_sharing_interval
            get_logger().info("use fixed sharing_interval %s", fixed_sharing_interval)
            return
        batch_number = self.config.algorithm_kwargs["batch_number"]
        if "init" in result.other_data:
            result.other_data["sharing_interval"] = self.last_interval
            return
        assert len(self.performance_stat) > 1
        if not self.is_training_slow():
            result.other_data["sharing_interval"] = self.last_interval
            return
        init_test_loss: float = self.performance_stat[0]["test_loss"]
        last_test_loss = self.performance_stat[max(self.performance_stat.keys())][
            "test_loss"
        ]
        interval: int = int(math.sqrt(last_test_loss / init_test_loss) * batch_number)
        if interval >= self.last_interval:
            interval = self.last_interval - 1
        min_sharing_interval = self.config.algorithm_kwargs.get(
            "min_sharing_interval", 1
        )
        interval = max(interval, min_sharing_interval)
        get_logger().warning(
            "init test loss is %s,last_interval is %s, interval is %s",
            init_test_loss,
            self.last_interval,
            interval,
        )
        self.last_interval = interval
        result.other_data["sharing_interval"] = interval

    def is_training_slow(self) -> bool:
        last_test_loss = self.performance_stat[max(self.performance_stat.keys())][
            "test_loss"
        ]
        last_to_second_test_loss = self.performance_stat[
            max(self.performance_stat.keys()) - 1
        ]["test_loss"]
        get_logger().warning(
            "last test loss is %s,last to second test loss is %s",
            last_test_loss,
            last_to_second_test_loss,
        )
        if last_test_loss >= last_to_second_test_loss:
            return True
        return False
