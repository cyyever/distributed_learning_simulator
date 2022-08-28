import datetime
import os
import typing
import uuid

import hydra
from cyy_torch_toolbox.default_config import DefaultConfig


class ExperimentConfig(DefaultConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distributed_algorithm: str = None
        self.worker_number = None
        self.parallel_number = None
        self.round = None
        self.iid = True
        self.distribute_init_parameters = True
        self.noise_percents: typing.Optional[list] = None
        self.log_file = None
        self.use_amp = False
        self.offload_memory = False
        self.endpoint_kwargs = {}
        self.algorithm_kwargs = {}
        self.frozen_modules = []

    def load_config_from_file(self, conf):
        DefaultConfig.load_config(self, conf)
        if self.iid:
            assert self.noise_percents is None
        else:
            if self.noise_percents is not None:
                assert isinstance(self.noise_percents, list)
                assert len(self.noise_percents) == self.worker_number
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
        if self.use_amp:
            trainer.set_amp()
        for module in self.frozen_modules:
            trainer.model_util.freeze_modules(module_name=module)
        return trainer


global_config = ExperimentConfig()


@hydra.main(config_path="conf", version_base=None)
def load_config(conf) -> None:
    if len(conf) == 1:
        conf = next(iter(conf.values()))
    global_config.load_config_from_file(conf)
