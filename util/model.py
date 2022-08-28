import torch.nn as nn
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.model_util import ModelUtil
from cyy_torch_toolbox.trainer import Trainer


def drop_normalization(model_util):
    get_logger().warning("remove BatchNorm")
    for module_type in (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d):
        model_util.change_modules(
            module_type=module_type,
            f=lambda name, _, __: model_util.register_module(name, nn.Identity()),
        )


def load_parameters(
    trainer: Trainer, parameter_dict: dict, reuse_learning_rate: bool
) -> None:
    if reuse_learning_rate:
        optimizer = trainer.get_optimizer()
        assert len(optimizer.param_groups) == 1
        old_param_group = optimizer.param_groups[0]
        optimizer.param_groups.clear()
        optimizer.state.clear()
        trainer.model_util.load_parameter_dict(parameter_dict)
        optimizer.add_param_group({"params": trainer.model.parameters()})
        for k, v in old_param_group.items():
            if k not in "params":
                optimizer.param_groups[0][k] = v
                get_logger().debug("reuse parameter property %s", k)
    else:
        trainer.load_parameter_dict(parameter_dict)
    trainer.model_util.reset_running_stats()


def get_module_blocks(
    model_util: ModelUtil,
    block_types: set = None,
) -> list:
    if block_types is None:
        block_types = {
            ("AlbertTransformer",),
            ("AlbertEmbeddings",),
            ("Bottleneck",),
            (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d),
            (nn.Conv2d, nn.BatchNorm2d, nn.ReLU),
            (nn.Conv2d, nn.ReLU, nn.MaxPool2d),
            (nn.BatchNorm2d, nn.ReLU, nn.Conv2d),
            (nn.BatchNorm2d, nn.Conv2d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv2d, nn.ReLU),
            (nn.Linear, nn.ReLU),
        }
    return model_util.get_module_blocks(block_types=block_types)
