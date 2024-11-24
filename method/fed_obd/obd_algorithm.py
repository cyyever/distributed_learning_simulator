import os
import sys

import torch
from cyy_naive_lib.log import log_debug, log_info
from cyy_torch_toolbox import ModelParameter
from cyy_torch_toolbox.tensor import cat_tensors_to_vector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from algorithm.block_algorithm import BlockAlgorithmMixin


class OpportunisticBlockDropoutAlgorithmMixin(BlockAlgorithmMixin):
    def __init__(self) -> None:
        super().__init__()
        self.__dropout_rate = self.config.algorithm_kwargs["dropout_rate"]
        log_debug("use dropout rate %s", self.__dropout_rate)
        self.__parameter_num: int = 0

    def get_block_parameter(self, parameter: ModelParameter) -> ModelParameter:
        if self.__parameter_num == 0:
            parameter_list = self.trainer.model_util.get_parameter_list()
            self.__parameter_num = len(parameter_list)
        threshold = (1 - self.__dropout_rate) * self.__parameter_num
        partial_parameter_num = 0
        new_parameter: dict = {}

        block_delta: dict = {}
        for block in self.blocks:
            block_dict, delta, block_size = self.__analyze_block(parameter, block)
            mean_delta = delta / block_size
            if mean_delta not in block_delta:
                block_delta[mean_delta] = []
            block_delta[mean_delta].append((block_dict, block_size))
        log_debug("block_delta is %s", sorted(block_delta.keys(), reverse=True))

        for mean_delta in sorted(block_delta.keys(), reverse=True):
            if partial_parameter_num > threshold:
                break
            for block_dict, block_size in block_delta[mean_delta]:
                if partial_parameter_num + block_size > threshold:
                    continue
                partial_parameter_num += block_size
                new_parameter |= block_dict
        log_debug("choose blocks %s", new_parameter.keys())
        log_info(
            "partial_parameter_num %s threshold %s parameter_num %s",
            partial_parameter_num,
            threshold,
            self.__parameter_num,
        )

        return new_parameter

    def __analyze_block(self, parameter: ModelParameter, block: list) -> tuple:
        cur_block_parameters = []
        prev_block_parameters = []
        block_dict = {}
        for submodule_name, submodule in block:
            for p_name, _ in submodule.named_parameters():
                parameter_name = submodule_name + "." + p_name
                cur_block_parameters.append(parameter[parameter_name])
                prev_block_parameters.append(self.model_cache.parameter[parameter_name])
                block_dict[parameter_name] = parameter[parameter_name]

        cur_block_parameter = cat_tensors_to_vector(cur_block_parameters)
        prev_block_parameter = cat_tensors_to_vector(prev_block_parameters)
        delta = torch.linalg.vector_norm(
            cur_block_parameter.cpu() - prev_block_parameter.cpu()
        ).item()
        return (block_dict, delta, cur_block_parameter.nelement())
