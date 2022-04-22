import os
import threading

import gevent
import gevent.lock
import torch
# from analysis.module_diff import ModuleDiff
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.device import get_cpu_device, put_data_to_device
from cyy_torch_toolbox.ml_type import ModelExecutorHookPoint
from cyy_torch_toolbox.trainer import Trainer
from executor import Executer


class Worker(Executer):
    semaphore = gevent.lock.BoundedSemaphore(value=1)
    __thread_data = threading.local()

    def __init__(self, config, worker_id: int, trainer: Trainer, endpoint, **kwargs):
        super().__init__(
            config=config, endpoint=endpoint, name=f"worker {worker_id}", **kwargs
        )
        self.__worker_id = worker_id
        self.trainer = trainer
        self._round_num = 0

    @property
    def worker_id(self):
        return self.__worker_id

    def __offload_from_gpu(self):
        self.trainer.offload_from_gpu()
        gpu_items = set()
        for attr, v in self.__dict__.items():
            has_cuda_tensor = False
            if torch.is_tensor(v):
                has_cuda_tensor = True
            elif isinstance(v, dict):
                for v2 in v.values():
                    if torch.is_tensor(v2):
                        has_cuda_tensor = True
                        break
            elif isinstance(v, (set, list, tuple)):
                for v2 in v:
                    if torch.is_tensor(v2):
                        has_cuda_tensor = True
                        break
            if has_cuda_tensor:
                gpu_items.add(attr)
        for attr in gpu_items:
            value = self.__dict__[attr]
            self.__dict__[attr] = put_data_to_device(value, get_cpu_device())

    def _release_semaphore(self):
        self.__offload_from_gpu()
        super()._release_semaphore()

    def before_training(self):
        pass

    def start(self, **kwargs):
        self.__offload_from_gpu()
        if self.__worker_id != 0:
            self.trainer.disable_logger()
        self.trainer.append_named_hook(
            ModelExecutorHookPoint.AFTER_BATCH,
            "release_device_lock",
            self._release_device_lock,
        )

        self.before_training()

        # if os.getenv("debug_diff"):
        #     self.trainer.append_hook(ModuleDiff())

        for self._round_num in range(1, self.config.round + 1):
            self._acquire_semaphore()
            if self._round_num == 1:
                self.trainer.set_device(self.get_device())
            self.trainer.batch_loss_logger.set_prefix(
                "round:" + str(self._round_num) + ","
            )
            self.trainer.performance_metric_logger.set_prefix(
                "round:" + str(self._round_num) + ","
            )
            self.trainer.visualizer.set_session_name(f"round_{self._round_num}")
            if os.getenv("use_best_model"):
                kwargs["keep_best_model"] = True
            self.trainer.train(**kwargs)
            self._release_semaphore()
        get_logger().debug("close endpoint")
        self._endpoint.close()
        get_logger().warning("finish worker %s", self.worker_id)
