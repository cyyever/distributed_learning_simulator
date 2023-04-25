""" FedDropoutAvg: Generalizable federated learning for histopathology image classification (https://arxiv.org/pdf/2111.13230.pdf) """

from typing import Any

from ..fed_avg_algorithm import FedAVGAlgorithm


class FedDropoutAvgAlgorithm(FedAVGAlgorithm):
    def _get_weight(self, dataset_size: int, name: str, parameter) -> Any:
        return (parameter != 0).float() * dataset_size

    def _adjust_total_weights(self, total_weights) -> Any:
        for k, v in total_weights.items():
            mask = (v == 0).bool()
            total_weights[k][mask] = 1
