import torch.nn as nn
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.model_util import ModelUtil


class DropNormalization(Hook):
    @staticmethod
    def identity(x):
        return x

    def _before_execute(self, **kwargs):
        model_executor = kwargs["model_executor"]
        get_logger().warning("remove BatchNorm from the network")
        model_util = ModelUtil(model_executor.model)
        for sub_module_type in (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d):
            model_util.change_sub_modules(
                sub_module_type=sub_module_type,
                f=lambda k, _: model_util.set_attr(
                    k, DropNormalization.identity, as_parameter=False
                ),
            )
