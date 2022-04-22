import copy
import json
import os

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.device import get_cpu_device, put_data_to_device

from .aggregation_algorithm import AggregationAlgorithm


class FedAVGAlgorithm(AggregationAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_model = None
        self.__stat = {}

    @property
    def prev_model(self):
        return self._prev_model

    def _process_worker_parameter(self, parameter):
        return parameter

    def _aggregate_worker_data_impl(self, worker_data):
        dataset_sizes = {}
        parameters = {}
        get_logger().info("begin aggregating %s", worker_data.keys())
        for worker_id, data in worker_data.items():
            training_dataset_size, parameter = data
            parameters[worker_id] = put_data_to_device(
                self._process_worker_parameter(parameter), device=get_cpu_device()
            )
            dataset_sizes[worker_id] = training_dataset_size
            get_logger().info("get worker data %s", worker_id)

        avg_parameter = AggregationAlgorithm.weighted_avg(
            parameters, AggregationAlgorithm.get_dataset_ratios(dataset_sizes)
        )
        return avg_parameter

    def _aggregate_worker_data(self, round_number, worker_data):
        avg_parameter = self._aggregate_worker_data_impl(worker_data)
        self._prev_model = copy.deepcopy(avg_parameter)
        metric = self.get_metric(self.prev_model)

        round_stat = {}
        round_stat["test_loss"] = metric["loss"]
        round_stat["test_acc"] = metric["acc"]

        get_logger().info(
            "round %s, test accuracy is %s", round_number, round_stat["test_acc"]
        )
        get_logger().info(
            "round %s, test loss is %s", round_number, round_stat["test_loss"]
        )

        self.__stat[round_number] = round_stat
        with open(
            os.path.join(self.save_dir, "round_stat.json"), "wt", encoding="utf-8"
        ) as f:
            json.dump(self.__stat, f)
        return avg_parameter
