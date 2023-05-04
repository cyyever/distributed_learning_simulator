""" Adaptive Federated Dropout: Improving Communication Efficiency and Generalization for Federated Learning (https://arxiv.org/abs/2011.04050)"""
import copy
import random

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.tensor import cat_tensors_to_vector

from ..fed_avg_algorithm import FedAVGAlgorithm


class SingleModelAdaptiveFedDropoutAlgorithm(FedAVGAlgorithm):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.__recorded: bool = False
        dropout_rate: float = config.algorithm_kwargs["dropout_rate"]
        assert dropout_rate < 1

        self.__score_map = {}
        self.__parameter_keys = set()
        self.__dropout_rate = dropout_rate
        self.__threshold = 0
        self.__l = 0
        self.__A = None

    def process_init_model(self, parameter_dict: dict) -> dict:
        self.__threshold = cat_tensors_to_vector(parameter_dict.values()).numel() * (
            1 - self.__dropout_rate
        )
        get_logger().warning("use threshold %s", self.__threshold)
        for k in parameter_dict:
            self.__score_map[k] = 0
        return {
            "partial_parameter": self.__random_selection(parameter_dict),
            "parameter": parameter_dict,
        }

    def __random_selection(self, parameter_dict: dict) -> dict:
        self.__parameter_keys = set()
        sub_parameter_dict = {}
        equal_prob = 1 / len(parameter_dict)
        accumulated_size = 0
        while accumulated_size < self.__threshold:
            for k, v in copy.copy(parameter_dict).items():
                if random.uniform(0, 1) <= equal_prob:
                    sub_parameter_dict[k] = v
                    accumulated_size += v.numel()
                    self.__parameter_keys.add(k)
                    parameter_dict.pop(k)
        get_logger().warning(
            "selection size %s, threshold %s", accumulated_size, self.__threshold
        )
        assert sub_parameter_dict
        self.__A = set(sub_parameter_dict.keys())
        return sub_parameter_dict

    def __weighted_random_selection(self, parameter_dict: dict) -> dict:
        # weighed random selection....
        self.__parameter_keys = set()
        sub_parameter_dict: dict = {}
        accumulated_size = 0
        score_map = copy.deepcopy(self.__score_map)
        # we sample from weighed parameters first
        score_map = {k: v for k, v in score_map.items() if v != 0}
        while accumulated_size < self.__threshold:
            for k in sub_parameter_dict:
                if k in score_map:
                    score_map.pop(k)
            if score_map:
                scale = sum(score_map.values())
                tmp_score_map = {k: v / scale for k, v in score_map.items()}
                for k, score in tmp_score_map.items():
                    if accumulated_size >= self.__threshold:
                        get_logger().warning(
                            "selection size %s, threshold %s",
                            accumulated_size,
                            self.__threshold,
                        )
                        return sub_parameter_dict
                    if random.uniform(0, 1) <= score:
                        v = parameter_dict[k]
                        sub_parameter_dict[k] = v
                        accumulated_size += v.numel()
                        self.__parameter_keys.add(k)
                        parameter_dict.pop(k)
                continue

            equal_prob = 1 / len(parameter_dict)
            for k, v in copy.copy(parameter_dict).items():
                if random.uniform(0, 1) <= equal_prob:
                    sub_parameter_dict[k] = v
                    accumulated_size += v.numel()
                    self.__parameter_keys.add(k)
                    parameter_dict.pop(k)
        assert sub_parameter_dict
        get_logger().warning(
            "selection size %s, threshold %s", accumulated_size, self.__threshold
        )
        return sub_parameter_dict

    def aggregate_worker_data(self) -> dict:
        result = super().aggregate_worker_data()
        training_loss_list = [
            data.data.pop("training_loss")
            for data in self._all_worker_data.values()
            if data is not None
        ]
        avg_training_loss = sum(training_loss_list) / len(training_loss_list)
        if avg_training_loss < self.__l:
            self.__recorded = True
            for k in self.__A:
                self.__score_map[k] = (self.__l - avg_training_loss) / self.__l
        else:
            self.__recorded = False
        self.__l = avg_training_loss
        if self.__recorded:
            result["partial_parameter"] = {k: result["parameter"][k] for k in self.__A}
            get_logger().info(
                "send_num %s",
                cat_tensors_to_vector(result["partial_parameter"].values()).numel(),
            )
            return result

        # weighed random selection....
        result["partial_parameter"] = self.__weighted_random_selection(
            result["parameter"]
        )
        self.__A = set(result["partial_parameter"].keys())
        get_logger().info(
            "send_num %s",
            cat_tensors_to_vector(result["partial_parameter"].values()).numel(),
        )
        return result
