import random

import torch
import torch.nn
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.tensor import cat_tensors_to_vector


class OpportunisticLayerDropoutAlgorithm:
    __print_blocks = False

    def __init__(self, dropout_rate: float):
        self.__dropout_rate = dropout_rate
        get_logger().warning("use dropout rate %s", self.__dropout_rate)
        self.__blocks = None
        self.__parameter_num = None

    def _find_blocks(self):
        self.__blocks = []
        modules = sorted(
            self.trainer.model_util.get_modules(), key=lambda x: -len(x[0])
        )
        added_modules = set()
        for submodule_name, submodule in modules:
            if not submodule_name:
                continue
            if len(list(submodule.parameters())) == 0:
                continue
            if any(a.startswith(submodule_name + ".") for a in added_modules):
                continue
            self.__blocks.append([(submodule_name, submodule)])
            added_modules.add(submodule_name)

        # check the parameter numbers are the same
        tmp_parameter_list = []
        tmp_parameter_name = set()
        for block in self.__blocks:
            for submodule_name, submodule in block:
                for p_name, p in submodule.named_parameters():
                    tmp_parameter_list.append(p)
                    if submodule_name:
                        tmp_parameter_name.add(submodule_name + "." + p_name)
                    else:
                        tmp_parameter_name.add(p_name)
        parameter_dict = self.trainer.model_util.get_parameter_dict()
        if tmp_parameter_name != set(parameter_dict.keys()):
            for a in tmp_parameter_name:
                if a not in parameter_dict:
                    raise RuntimeError(a + " not in model")
            for a in parameter_dict:
                if a not in tmp_parameter_name:
                    raise RuntimeError(a + " not in block")
        parameter_list = self.trainer.model_util.get_parameter_list()
        assert cat_tensors_to_vector(tmp_parameter_list).shape == parameter_list.shape
        self.__parameter_num = len(parameter_list)

    def get_block_parameter(self, parameter_dict: dict) -> dict:
        threshold = (1 - self.__dropout_rate) * self.__parameter_num
        partial_parameter_num = 0
        new_parameter_dict: dict = {}
        random.shuffle(self.__blocks)
        for block in self.__blocks:
            if partial_parameter_num > threshold:
                break
            block_dict = {}
            block_size = 0
            for submodule_name, submodule in block:
                for p_name, p in submodule.named_parameters():
                    parameter_name = submodule_name + "." + p_name
                    block_dict[parameter_name] = p
                    block_size += p.nelement()
            if partial_parameter_num + block_size > threshold:
                continue
            partial_parameter_num += block_size
            new_parameter_dict |= block_dict
        get_logger().info("choose blocks %s", new_parameter_dict.keys())
        get_logger().info(
            "partial_parameter_num %s threshold %s",
            partial_parameter_num,
            threshold,
        )
        return new_parameter_dict

    def __analyze_block(self, parameter_dict, block) -> tuple:
        cur_block_parameters = []
        prev_block_parameters = []
        block_dict = {}
        for submodule_name, submodule in block:
            for p_name, _ in submodule.named_parameters():
                parameter_name = submodule_name + "." + p_name
                cur_block_parameters.append(parameter_dict[parameter_name])
                prev_block_parameters.append(
                    self._model_cache.get_parameter_dict[parameter_name]
                )
                block_dict[parameter_name] = parameter_dict[parameter_name]

        cur_block_parameter = cat_tensors_to_vector(cur_block_parameters)
        prev_block_parameter = cat_tensors_to_vector(prev_block_parameters)
        delta = torch.linalg.norm(
            cur_block_parameter.cpu() - prev_block_parameter.cpu()
        ).item()
        return (block_dict, delta, cur_block_parameter.nelement())
