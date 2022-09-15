import torch
import torch.nn
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.tensor import cat_tensors_to_vector
from util.model import get_module_blocks


class OpportunisticBlockDropoutAlgorithm:
    __dropout_rate = None
    __blocks = None
    __parameter_num = None

    def _find_blocks(self):
        self.__dropout_rate = self.config.algorithm_kwargs["dropout_rate"]
        get_logger().warning("use dropout rate %s", self.__dropout_rate)
        self.__blocks = get_module_blocks(self.trainer.model_util)

        if self.worker_id == 0:
            get_logger().info("identify these blocks in model:")
            for block in self.__blocks:
                get_logger().info(
                    "%s",
                    [f"{name}" for name, _ in block],
                )

        for submodule_name, submodule in self.trainer.model_util.get_modules():
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
                if self.worker_id == 0:
                    get_logger().info("identify a submodule:%s", submodule_name)

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

    def get_block_parameter(self, parameter_dict) -> tuple[dict, dict]:
        block_delta: dict = {}
        for block in self.__blocks:
            block_dict, delta, block_size = self.__analyze_block(parameter_dict, block)
            mean_delta = delta / block_size
            if mean_delta not in block_delta:
                block_delta[mean_delta] = []
            block_delta[mean_delta].append((block_dict, block_size))
        get_logger().info("block_delta is %s", sorted(block_delta.keys(), reverse=True))
        partial_parameter_num = 0
        new_parameter_dict: dict = {}

        threshold = (1 - self.__dropout_rate) * self.__parameter_num
        for mean_delta in sorted(block_delta.keys(), reverse=True):
            if partial_parameter_num > threshold:
                break
            for (block_dict, block_size) in block_delta[mean_delta]:
                if partial_parameter_num + block_size > threshold:
                    continue
                partial_parameter_num += block_size
                new_parameter_dict |= block_dict
        get_logger().debug("choose blocks %s", new_parameter_dict.keys())
        get_logger().info(
            "partial_parameter_num %s threshold %s", partial_parameter_num, threshold
        )

        remain_parameter_dict = {}
        for k, v in parameter_dict.items():
            if k not in new_parameter_dict:
                remain_parameter_dict[k] = v - self.cached_parameter_dict[k]
        get_logger().debug("remain_parameter_dict are %s", remain_parameter_dict.keys())
        assert len(new_parameter_dict) + len(remain_parameter_dict) == len(
            parameter_dict
        )
        return new_parameter_dict, remain_parameter_dict

    def __analyze_block(self, parameter_dict, block) -> tuple:
        cur_block_parameters = []
        prev_block_parameters = []
        block_dict = {}
        for submodule_name, submodule in block:
            for p_name, _ in submodule.named_parameters():
                parameter_name = submodule_name + "." + p_name
                cur_block_parameters.append(parameter_dict[parameter_name])
                prev_block_parameters.append(self.cached_parameter_dict[parameter_name])
                block_dict[parameter_name] = parameter_dict[parameter_name]

        cur_block_parameter = cat_tensors_to_vector(cur_block_parameters)
        prev_block_parameter = cat_tensors_to_vector(prev_block_parameters)
        delta = torch.linalg.norm(
            cur_block_parameter.cpu() - prev_block_parameter.cpu()
        ).item()
        return (block_dict, delta, cur_block_parameter.nelement())
