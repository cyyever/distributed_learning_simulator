import os
from typing import Any

from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.model_executor import ModelExecutor
from model_transform.drop_normalization import DropNormalization


class AggregationAlgorithm:
    __tester = None

    @property
    def tester(self) -> ModelExecutor:
        if self.__tester is None:
            self._acquire_semaphore()
            self.__tester = self.config.create_inferencer(
                phase=MachineLearningPhase.Test
            )
            self.__tester.disable_logger()
            if os.getenv("drop_norm"):
                self.__tester.append_hook(DropNormalization())
            self.__tester.set_device(self.get_device())
            self._release_semaphore()
        return self.__tester

    def get_metric(self, parameter_dict):
        model_util = self.tester.model_util
        model_util.load_parameter_dict(parameter_dict)
        model_util.remove_statistical_variables()
        self.tester.inference(epoch=1)
        metric = {
            "acc": self.tester.performance_metric.get_accuracy(1).item(),
            "loss": self.tester.performance_metric.get_loss(1).item(),
        }
        self.tester.offload_from_gpu()
        return metric

    def _aggregate_worker_data(self, worker_data: dict):
        raise NotImplementedError()

    @classmethod
    def get_dataset_ratios(cls, dataset_sizes: dict) -> dict:
        total_training_dataset_size = sum(dataset_sizes.values())
        return {
            k: float(v) / float(total_training_dataset_size)
            for (k, v) in dataset_sizes.items()
        }

    @classmethod
    def weighted_avg(cls, data_dict: dict, weight_dict: dict) -> Any:
        avg_data = None
        for k, d in data_dict.items():
            ratio = weight_dict[k]
            assert 0 <= ratio <= 1

            if isinstance(d, dict):
                d = {k2: v2 * ratio for (k2, v2) in d.items()}
                if avg_data is None:
                    avg_data = d
                else:
                    for k in avg_data:
                        avg_data[k] += d[k]
            else:
                d = d * ratio
                if avg_data is None:
                    avg_data = d
                else:
                    avg_data += d
        return avg_data
