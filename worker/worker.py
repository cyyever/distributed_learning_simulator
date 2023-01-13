from typing import Any

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import ModelExecutorHookPoint
from cyy_torch_toolbox.trainer import Trainer
from executor import Executor


class Worker(Executor):
    def __init__(
        self,
        task_id: Any,
        worker_id: int,
        trainer: Trainer,
        **kwargs: dict,
    ):
        name = f"worker {worker_id}"
        if task_id is not None:
            name = f"worker {worker_id} of {task_id}"
        super().__init__(name=name, **kwargs)
        self.__worker_id = worker_id
        self.__trainer = trainer
        self._round_num = 0
        self._force_stop = False

    @property
    def worker_id(self):
        return self.__worker_id

    @property
    def trainer(self):
        return self.__trainer

    def _offload_from_memory(self):
        if self.config.offload_memory:
            self.__trainer.offload_from_gpu()

    def _before_training(self):
        self._force_stop = False
        self.trainer.set_device(
            self._get_device(
                lock_callback=lambda: self.trainer.append_named_hook(
                    ModelExecutorHookPoint.AFTER_BATCH,
                    "release_device_lock",
                    self._release_device_lock,
                )
            )
        )

    def _stopped(self) -> bool:
        return self._round_num > self.config.round or self._force_stop

    def start(self, **kwargs):
        first_training: bool = True
        self._round_num = 1
        while not self._stopped():
            # in case worker changes round number
            with self._get_context():
                if first_training:
                    self._before_training()
                    first_training = False
                    # in case worker changes round number
                    if self._stopped():
                        break
                if not first_training:
                    self.trainer.disable_hook("logger")
                self.trainer.set_visualizer_prefix(f"round: {self._round_num},")
                self.trainer.set_save_dir(self.save_dir)
                self.trainer.train(
                    **kwargs,
                    batch_loss_log_times=None if self.config.log_batch_loss else 0,
                )
                self._offload_from_memory()
                self._round_num += 1
        get_logger().debug("close endpoint")
        self._endpoint.close()
        get_logger().debug("finish worker %s", self.worker_id)
