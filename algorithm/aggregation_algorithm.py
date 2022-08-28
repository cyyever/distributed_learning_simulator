import copy
from typing import Any

from cyy_naive_lib.log import get_logger
from cyy_naive_lib.storage import DataStorage


class AggregationAlgorithm:
    @classmethod
    def get_ratios(cls, scalars: dict) -> dict:
        total_scalar = sum(scalars.values())
        return {k: float(v) / float(total_scalar) for (k, v) in scalars.items()}

    @classmethod
    def weighted_avg(cls, data_dict: dict, weight_dict: dict) -> Any:
        avg_data = None
        for k, v in data_dict.items():
            ratio = weight_dict[k]
            assert 0 <= ratio <= 1
            d = v
            if isinstance(v, DataStorage):
                d = v.data

            match d:
                case dict():
                    d = {k2: v2 * ratio for (k2, v2) in d.items()}
                case _:
                    d = d * ratio
            if isinstance(v, DataStorage):
                v.save()
            if avg_data is None:
                avg_data = d
            else:
                if isinstance(avg_data, dict):
                    for k in avg_data:
                        avg_data[k] += d[k]
                else:
                    avg_data += d
        return avg_data

    def extract_data(
        self, worker_data: dict[str, DataStorage], old_parameter_dict: dict | None
    ) -> dict:
        dataset_sizes = {}
        parameters = {}
        get_logger().debug("begin aggregating %s", worker_data.keys())
        for worker_id, data in worker_data.items():
            data = data.data
            dataset_sizes[worker_id] = data["dataset_size"]
            if "use_distributed_model" in data:
                assert old_parameter_dict is not None
                parameters[worker_id] = copy.deepcopy(old_parameter_dict)
            elif "parameter" in data:
                parameters[worker_id] = data["parameter"]
            else:
                assert old_parameter_dict is not None
                parameters[worker_id] = copy.deepcopy(old_parameter_dict)
                if "parameter_diff" not in data:
                    data["parameter_diff"] = {}
                if "quantized_parameter_diff" in data:
                    get_logger().debug("process quantized_parameter_diff")
                    for k, v in data["quantized_parameter_diff"].items():
                        assert k not in data["parameter_diff"]
                        data["parameter_diff"][k] = v

                for k, v in data["parameter_diff"].items():
                    parameters[worker_id][k] += v
            get_logger().debug("get worker data %s", worker_id)
        return {"parameter": parameters, "dataset_size": dataset_sizes}
