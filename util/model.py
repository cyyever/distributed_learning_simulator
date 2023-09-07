from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.trainer import Trainer


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
    trainer.model_util.disable_running_stats()
