import datetime
import os
import uuid

import hydra
import omegaconf
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.device import get_devices
from hydra.core.hydra_config import HydraConfig


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
        dir_suffix = os.path.join(
            self.distributed_algorithm + log_suffix
            if self.iid
            else self.distributed_algorithm + "_non_iid" + log_suffix,
            self.dc_config.dataset_name,
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


@hydra.main(config_path="conf", version_base=None)
def load_config(conf) -> None:
    conf = next(iter(conf.values()))
    global_conf_path = os.path.join(
        HydraConfig.get().runtime.cwd, "conf", "global.yaml"
    )
    result_conf = omegaconf.OmegaConf.load(global_conf_path)
    result_conf.merge_with(conf)
    global_config.load_config_and_process(result_conf)
