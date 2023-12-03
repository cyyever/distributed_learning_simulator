import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.model.util import ModelUtil
from cyy_torch_toolbox.tensor import cat_tensors_to_vector
from torch import nn


def get_module_blocks(
    model_util: ModelUtil,
    block_types: set | None = None,
) -> list:
    if block_types is None:
        block_types = {
            ("AlbertTransformer",),
            ("AlbertEmbeddings",),
            ("Bottleneck",),
            ("TransformerEncoderLayer",),
            (nn.BatchNorm2d, nn.ReLU, nn.Conv2d),
            (nn.BatchNorm2d, nn.Conv2d),
            (nn.Conv2d, nn.BatchNorm2d),
        }
    return model_util.get_module_blocks(block_types=block_types)


class OpportunisticBlockDropoutAlgorithm:
    __print_blocks = False

    def __init__(self, dropout_rate: float):
        self.__dropout_rate = dropout_rate
        get_logger().warning("use dropout rate %s", self.__dropout_rate)
        self.__blocks: list | None = None
        self.__parameter_num = None

    def __find_blocks(self, model_util: ModelUtil) -> None:
        self.__blocks = get_module_blocks(model_util)
        for submodule_name, submodule in model_util.get_modules():
            if not submodule_name:
                continue
            if len(list(submodule.parameters())) == 0:
                continue
            remain = True
            for block in self.__blocks:
                for block_submodule_name, _ in block:
                    if (
                        block_submodule_name == submodule_name
                        or submodule_name.startswith(block_submodule_name + ".")
                        or block_submodule_name.startswith(submodule_name + ".")
                    ):
                        remain = False
                        break
                if not remain:
                    break
            if remain:
                self.__blocks.append([(submodule_name, submodule)])
                if not self.__print_blocks:
                    get_logger().info("identify a submodule:%s", submodule_name)

        if not self.__print_blocks:
            OpportunisticBlockDropoutAlgorithm.__print_blocks = False
            get_logger().info("identify these blocks in model:")
            for block in self.__blocks:
                get_logger().info(
                    "%s",
                    [f"{name}" for name, _ in block],
                )

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
        parameter_dict = model_util.get_parameter_dict()
        if tmp_parameter_name != set(parameter_dict.keys()):
            for a in tmp_parameter_name:
                if a not in parameter_dict:
                    raise RuntimeError(a + " not in model")
            for a in parameter_dict:
                if a not in tmp_parameter_name:
                    raise RuntimeError(a + " not in block")
        parameter_list = model_util.get_parameter_list()
        assert cat_tensors_to_vector(tmp_parameter_list).shape == parameter_list.shape
        self.__parameter_num = len(parameter_list)

    def get_block_parameter(
        self, parameter_dict: dict, model_util: ModelUtil, model_cache
    ) -> dict:
        if self.__blocks is None:
            self.__find_blocks(model_util=model_util)
        threshold = (1 - self.__dropout_rate) * self.__parameter_num
        partial_parameter_num = 0
        new_parameter_dict: dict = {}

        block_delta: dict = {}
        for block in self.__blocks:
            block_dict, delta, block_size = self.__analyze_block(
                parameter_dict, block, model_cache
            )
            mean_delta = delta / block_size
            if mean_delta not in block_delta:
                block_delta[mean_delta] = []
            block_delta[mean_delta].append((block_dict, block_size))
        get_logger().debug(
            "block_delta is %s", sorted(block_delta.keys(), reverse=True)
        )

        for mean_delta in sorted(block_delta.keys(), reverse=True):
            if partial_parameter_num > threshold:
                break
            for block_dict, block_size in block_delta[mean_delta]:
                if partial_parameter_num + block_size > threshold:
                    continue
                partial_parameter_num += block_size
                new_parameter_dict |= block_dict
        get_logger().debug("choose blocks %s", new_parameter_dict.keys())
        get_logger().info(
            "partial_parameter_num %s threshold %s parameter_num %s",
            partial_parameter_num,
            threshold,
            self.__parameter_num,
        )

        return new_parameter_dict

    def __analyze_block(self, parameter_dict, block, model_cache) -> tuple:
        cur_block_parameters = []
        prev_block_parameters = []
        block_dict = {}
        for submodule_name, submodule in block:
            for p_name, _ in submodule.named_parameters():
                parameter_name = submodule_name + "." + p_name
                cur_block_parameters.append(parameter_dict[parameter_name])
                prev_block_parameters.append(model_cache.parameter_dict[parameter_name])
                block_dict[parameter_name] = parameter_dict[parameter_name]

        cur_block_parameter = cat_tensors_to_vector(cur_block_parameters)
        prev_block_parameter = cat_tensors_to_vector(prev_block_parameters)
        delta = torch.linalg.norm(
            cur_block_parameter.cpu() - prev_block_parameter.cpu()
        ).item()
        return (block_dict, delta, cur_block_parameter.nelement())
