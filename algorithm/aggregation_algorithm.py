import copy
import os
from typing import Any

from cyy_naive_lib.storage import DataStorage
from cyy_torch_toolbox.device import get_cpu_device
from cyy_torch_toolbox.tensor import tensor_to


class AggregationAlgorithm:
    def __init__(self, server=None) -> None:
        self._all_worker_data = {}
        self._server = server

    def process_init_model(self, parameter_dict):
        return {"parameter": parameter_dict}

    @classmethod
    def get_ratios(cls, data_dict: dict, key_name: str) -> dict:
        total_scalar = sum(v.data[key_name] for v in data_dict.values())
        return {
            k: float(v.data[key_name]) / float(total_scalar)
            for k, v in data_dict.items()
        }

    @classmethod
    def weighted_avg(cls, data_dict: dict, weight_dict: dict, key_name: str) -> Any:
        avg_data = None
        for k, v in data_dict.items():
            ratio = weight_dict[k]
            assert 0 <= ratio <= 1
            d = v.data[key_name]

            match d:
                case dict():
                    d = {k2: v2 * ratio for (k2, v2) in d.items()}
                case _:
                    d = d * ratio
            if avg_data is None:
                avg_data = d
            else:
                if isinstance(avg_data, dict):
                    for k in avg_data:
                        avg_data[k] += d[k]
                else:
                    avg_data += d
        return avg_data

    def __process_worker_data(
        self,
        worker_data: dict[str, DataStorage],
        old_parameter_dict: dict | None,
    ) -> dict:
        res = worker_data
        if "use_distributed_model" in res:
            assert old_parameter_dict is not None
            res["parameter"] = copy.deepcopy(old_parameter_dict)
        assert not ("parameter_diff" in res and "parameter" in res)
        if "parameter_diff" in res:
            assert old_parameter_dict is not None
            res["parameter"] = copy.deepcopy(old_parameter_dict)
            for k, v in res["parameter_diff"].items():
                res["parameter"][k] += v
            res.pop("parameter_diff")
        if "parameter" in res and old_parameter_dict is not None:
            for k, v in old_parameter_dict.items():
                if k not in res["parameter"]:
                    res["parameter"][k] = v
        return res

    def process_worker_data(
        self,
        worker_id,
        worker_data: dict[str, DataStorage],
        old_parameter_dict: dict | None,
        save_dir: str,
    ) -> None:
        if worker_data is None:
            self._all_worker_data[worker_id] = worker_data
            return
        worker_data = self.__process_worker_data(
            worker_data=tensor_to(worker_data, device=get_cpu_device()),
            old_parameter_dict=old_parameter_dict,
        )
        worker_data = DataStorage(
            data=worker_data,
            data_path=os.path.join(save_dir, "worker_data", str(worker_id)),
        )
        self._all_worker_data[worker_id] = worker_data

    def aggregate_worker_data(self) -> dict:
        raise NotImplementedError()
