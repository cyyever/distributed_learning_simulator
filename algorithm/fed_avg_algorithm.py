from typing import Any

import torch
from cyy_naive_lib.storage import DataStorage

from .aggregation_algorithm import AggregationAlgorithm


class FedAVGAlgorithm(AggregationAlgorithm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.accumulate: bool = True
        self.__dtypes: dict = {}
        self.__total_weights: dict = {}
        self.__parameter: dict = {}

    def process_worker_data(
        self,
        worker_id: int,
        worker_data: dict[str, DataStorage],
        old_parameter_dict: dict | None,
        save_dir: str,
    ) -> None:
        super().process_worker_data(
            worker_id=worker_id,
            worker_data=worker_data,
            old_parameter_dict=old_parameter_dict,
            save_dir=save_dir,
        )
        if not self.accumulate:
            return
        if self._all_worker_data[worker_id] is None:
            return
        if "dataset_size" not in self._all_worker_data[worker_id].data:
            return
        dataset_size = self._all_worker_data[worker_id].data["dataset_size"]
        for k, v in self._all_worker_data[worker_id].data["parameter"].items():
            self.__dtypes[k] = v.dtype
            weight = self._get_weight(dataset_size=dataset_size, name=k, parameter=v)
            tmp = v.to(dtype=torch.float64) * weight
            if k not in self.__parameter:
                self.__parameter[k] = tmp
            else:
                self.__parameter[k] += tmp
            self.__total_weights = self._accumulate_weight(
                total_weights=self.__total_weights, weight=weight, name=k, parameter=v
            )
        self._all_worker_data[worker_id].data["parameter"] = None

    def _get_weight(self, dataset_size, name, parameter: dict) -> Any:
        return dataset_size

    def _accumulate_weight(self, total_weights, weight, name, parameter) -> Any:
        if name not in total_weights:
            total_weights[name] = weight
        else:
            total_weights[name] += weight
        return total_weights

    def _adjust_total_weights(self, total_weights) -> Any:
        pass

    def aggregate_worker_data(self) -> dict:
        res = {}
        if not self.accumulate:
            if "parameter" in next(iter(self._all_worker_data.values())).data:
                parameter = AggregationAlgorithm.weighted_avg(
                    self._all_worker_data,
                    AggregationAlgorithm.get_ratios(
                        self._all_worker_data, key_name="dataset_size"
                    ),
                    key_name="parameter",
                )
                res = {"parameter": parameter}
        else:
            if self.__parameter:
                parameter = self.__parameter
                self.__parameter = {}
                self._adjust_total_weights(self.__total_weights)
                for k, v in parameter.items():
                    parameter[k] = (v / self.__total_weights[k]).to(
                        dtype=self.__dtypes[k]
                    )
                self.__total_weights = {}
                res = {"parameter": parameter}

        for worker_data in self._all_worker_data.values():
            if worker_data is None:
                continue
            for k, v in worker_data.data.items():
                if k not in ["parameter", "dataset_size"]:
                    res[k] = v
            break
        return res
