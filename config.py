import datetime
import os
import uuid

import hydra
import omegaconf
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.device import get_devices


class DistributedTrainingConfig(DefaultConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distributed_algorithm: str = ""
        self.worker_number: int = 0
        self.parallel_number: int = len(get_devices())
        self.round: int = 0
        self.iid: bool = True
        self.log_batch_loss: bool = False
        self.distribute_init_parameters: bool = True
        self.server_send_file: bool = False
        self.log_file: None | str = None
        self.offload_memory: bool = False
        self.endpoint_kwargs: dict = {}
        self.algorithm_kwargs: dict = {}
        self.frozen_modules: list = []

    def load_config_and_process(self, conf) -> None:
        DefaultConfig.load_config(self, conf)
        task_time = datetime.datetime.now()
        date_time = "{date:%Y-%m-%d_%H_%M_%S}".format(date=task_time)
        log_suffix = self.algorithm_kwargs.get("log_suffix", "")
        dataset_name = self.dc_config.dataset_kwargs.get(
            "name", self.dc_config.dataset_name
        )
        dir_suffix = os.path.join(
            self.distributed_algorithm + log_suffix
            if self.iid
            else self.distributed_algorithm + "_non_iid" + log_suffix,
            dataset_name,
            self.model_config.model_name,
            date_time,
            str(uuid.uuid4()),
        )
        self.save_dir = os.path.join("session", dir_suffix)
        self.log_file = str(os.path.join("log", dir_suffix)) + ".log"

    def create_trainer(self, *args, **kwargs):
        trainer = super().create_trainer(*args, **kwargs)
        for module in self.frozen_modules:
            trainer.model_util.freeze_modules(module_name=module)
        return trainer


global_config: DistributedTrainingConfig = DistributedTrainingConfig()


def __load_config(conf) -> None:
    global_conf_path = os.path.join(os.path.dirname(__file__), "conf", "global.yaml")
    result_conf = omegaconf.OmegaConf.load(global_conf_path)
    result_conf.merge_with(conf)
    global_config.load_config_and_process(result_conf)


@hydra.main(config_path="conf", version_base=None)
def load_config(conf) -> None:
    while "dataset_name" not in conf and len(conf) == 1:
        conf = next(iter(conf.values()))
    __load_config(conf)


def load_config_from_file(
    dataset_name: str | None = None,
    distributed_algorithm: str | None = None,
    config_file: None | str = None,
) -> DistributedTrainingConfig:
    if config_file is None:
        config_file = os.path.join(
            os.path.dirname(__file__),
            "conf",
            distributed_algorithm,
            dataset_name + ".yaml",
        )
    conf = omegaconf.OmegaConf.load(config_file)
    __load_config(conf)
    return global_config
