""" FedDropoutAvg: Generalizable federated learning for histopathology image classification (https://arxiv.org/pdf/2111.13230.pdf) """
import copy

import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.storage import DataStorage
from cyy_torch_toolbox.tensor import cat_tensors_to_vector, load_tensor_dict

from .aggregation_algorithm import AggregationAlgorithm


class FedDropoutAvgAlgorithm(AggregationAlgorithm):
    def aggregate_worker_data(
        self, worker_data: dict[str, DataStorage], old_parameter_dict: dict | None
    ) -> dict:
        parameter_list = {}
        weights = {}
        get_logger().debug("begin aggregating %s", worker_data.keys())
        total_weight: torch.Tensor = None
        shapes = None
        for worker_id, data in worker_data.items():
            data = data.data
            parameter_list[worker_id] = cat_tensors_to_vector(
                get_mapping_values_by_key_order(data["parameter"])
            )
            weights[worker_id] = (parameter_list[worker_id] != 0).float() * data[
                "dataset_size"
            ]
            if total_weight is None:
                total_weight = copy.deepcopy(weights[worker_id])
            else:
                total_weight += weights[worker_id]
            if shapes is None:
                shapes = {k: v.shape for k, v in data["parameter"].items()}
        # avoid Division by zero
        mask = total_weight == 0
        for v in weights.values():
            v[mask] = 0
        total_weight[mask] = 1
        final_parameter_list = None
        for k, v in parameter_list.items():
            res = v * weights[k] / total_weight
            if final_parameter_list is None:
                final_parameter_list = res
            else:
                final_parameter_list += res
        return load_tensor_dict(shapes=shapes, tensor=final_parameter_list)
