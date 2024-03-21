from typing import Any

import torch

from ..common_import import FedAVGAlgorithm


class FedDropoutAvgAlgorithm(FedAVGAlgorithm):
    def _get_weight(self, worker_data, name: str, parameter: torch.Tensor) -> Any:
        weight = super()._get_weight(
            worker_data=worker_data, name=name, parameter=parameter
        )
        return (parameter != 0).float() * weight

    def _apply_total_weight(
        self, name: str, parameter: torch.Tensor, total_weight: Any
    ) -> torch.Tensor:
        # avoid dividing by zero, in such case we set weight to 1
        total_weight[total_weight == 0] = 1
        return super()._apply_total_weight(
            name=name, parameter=parameter, total_weight=total_weight
        )
