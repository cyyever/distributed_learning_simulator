import datetime
import os
import uuid

import hydra
from cyy_torch_toolbox.default_config import DefaultConfig

# from omegaconf import OmegaConf


class DistributedTrainingConfig(DefaultConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distributed_algorithm: str = None
        self.worker_number: None | int = None
        self.parallel_number: None | int = None
        self.round: None | int = None
        self.iid: bool = True
        self.distribute_init_parameters: bool = True
        # self.noise_percents: typing.Optional[list] = None
        self.log_file: None | str = None
        self.offload_memory: bool = False
        self.endpoint_kwargs: dict = {}
        self.algorithm_kwargs: dict = {}
        self.frozen_modules: list = []

    def load_config_from_file(self, conf) -> None:
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
    global_config.load_config_from_file(conf)


# def load_config_from_file(dataset_name: str, distributed_algorithm: str):
#     config_path = (
#         os.path.join(
#             os.path.dirname(os.path.realpath(__file__)),
#             "conf",
#             distributed_algorithm,
#             dataset_name,
#         )
#         + ".yaml"
#     )
#     conf = OmegaConf.load(config_path)
#     global_config.load_config_from_file(conf)
#     return global_config
