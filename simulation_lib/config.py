import copy
import datetime
import os
import uuid
from typing import Any

import hydra
import omegaconf
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox import Config
from cyy_torch_toolbox.dataset import ClassificationDatasetCollection
from cyy_torch_toolbox.device import get_device_memory_info

from .dependency import import_results  # noqa: F401
from .practitioner import Practitioner
from .sampler import get_dataset_collection_split


class DistributedTrainingConfig(Config):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.distributed_algorithm: str = ""
        self.algorithm_kwargs: dict = {}
        self.worker_number: int = 0
        self.round: int = 0
        self.dataset_sampling: str = "iid"
        self.dataset_sampling_kwargs: dict[str, Any] = {}
        self.endpoint_kwargs: dict = {}
        self.exp_name: str = ""
        self.merge_validation_to_training_set = False
        self.log_file: str = ""
        self.enable_training_log: bool = False
        self.worker_number_per_process: int = 0

    def load_config_and_process(self, conf: Any) -> None:
        self.load_config(conf)
        self.reset_session()

    def get_worker_number_per_process(self) -> int:
        if self.worker_number_per_process != 0:
            return self.worker_number_per_process

        memory_info = get_device_memory_info()
        refined_memory_info: dict = {}
        MB = 1024 * 1024
        GB = MB * 1024
        for device, info in memory_info.items():
            if info.used / info.total > 0.95:
                continue
            free_GB = int(info.free / GB)
            if free_GB == 0:
                continue
            refined_memory_info[device] = info.free
        assert refined_memory_info
        if self.worker_number <= len(refined_memory_info):
            return 1
        # small scale training
        if self.worker_number <= 50:
            return int(self.worker_number / len(refined_memory_info))
        total_bytes = sum(refined_memory_info.values())
        MB_per_worker = min(total_bytes / MB / self.worker_number, 10 * GB)
        get_logger().debug(
            "MB_per_worker %s other %s",
            MB_per_worker,
            min(refined_memory_info.values()) / MB,
        )
        worker_number_per_process = int(
            min(refined_memory_info.values()) / MB / MB_per_worker
        )
        assert worker_number_per_process > 0
        return worker_number_per_process

    def reset_session(self) -> None:
        task_time = datetime.datetime.now()
        date_time = f"{task_time:%Y-%m-%d_%H_%M_%S}"
        dataset_name = self.dc_config.dataset_kwargs.get(
            "name", self.dc_config.dataset_name
        )
        dir_suffix = os.path.join(
            self.distributed_algorithm,
            (
                f"{dataset_name}_{self.dataset_sampling}"
                if isinstance(self.dataset_sampling, str)
                else f"{dataset_name}_{'_'.join(self.dataset_sampling)}"
            ),
            self.model_config.model_name,
            date_time,
            str(uuid.uuid4().int + os.getpid()),
        )
        if self.exp_name:
            dir_suffix = os.path.join(self.exp_name, dir_suffix)
        self.save_dir = os.path.join("session", dir_suffix)
        self.log_file = str(os.path.join("log", dir_suffix)) + ".log"

    def create_practitioners(self) -> set:
        practitioners = set()
        dataset_collection = self.create_dataset_collection()
        assert isinstance(dataset_collection, ClassificationDatasetCollection)
        sampler = get_dataset_collection_split(
            name=self.dataset_sampling,
            dataset_collection=dataset_collection,
            part_number=self.worker_number,
            **self.dataset_sampling_kwargs,
        )
        for practitioner_id in range(self.worker_number):
            practitioner = Practitioner(
                practitioner_id=practitioner_id,
            )
            practitioner.set_sampler(sampler=sampler)
            practitioners.add(practitioner)
        assert practitioners
        return practitioners


global_config: DistributedTrainingConfig = DistributedTrainingConfig()


def __load_config(conf) -> None:
    global_conf_path = os.path.join(
        os.path.dirname(__file__), "..", "conf", "global.yaml"
    )
    if not os.path.isfile(global_conf_path):
        global_conf_path = os.path.join(
            os.path.dirname(__file__), "conf", "global.yaml"
        )
    result_conf = omegaconf.OmegaConf.load(global_conf_path)
    result_conf.merge_with(conf)
    global_config.load_config_and_process(result_conf)


@hydra.main(config_path="../conf", version_base=None)
def load_config(conf) -> None:
    while "dataset_name" not in conf and len(conf) == 1:
        conf = next(iter(conf.values()))
    __load_config(conf)


def load_config_from_file(
    config_file: None | str = None,
) -> DistributedTrainingConfig:
    assert config_file is not None
    conf = omegaconf.OmegaConf.load(config_file)
    __load_config(conf)
    return copy.deepcopy(global_config)
