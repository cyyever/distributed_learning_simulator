import random

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.tensor import cat_tensors_to_vector


class RandomDropoutAlgorithm:
    def __init__(self, dropout_rate):
        self.__dropout_rate = dropout_rate
        get_logger().warning("use dropout rate %s", self.__dropout_rate)

    def drop_parameter(self, parameter_dict: dict) -> dict:
        parameter_num = cat_tensors_to_vector(parameter_dict.values()).nelement()
        threshold = (1 - self.__dropout_rate) * parameter_num
        partial_parameter_num = 0
        parameter_names = list(parameter_dict.keys())
        random.shuffle(parameter_names)
        new_parameter_dict: dict = {}
        for k in parameter_names:
            if partial_parameter_num > threshold:
                break
            parameter = parameter_dict[k]
            if partial_parameter_num + parameter.nelement() > threshold:
                continue
            new_parameter_dict[k] = parameter
        get_logger().info(
            "partial_parameter_num %s threshold %s",
            partial_parameter_num,
            threshold,
        )
        return new_parameter_dict
