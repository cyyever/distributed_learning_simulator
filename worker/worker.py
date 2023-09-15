from functools import cached_property
from typing import Any

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import ExecutorHookPoint
from cyy_torch_toolbox.trainer import Trainer
from executor import Executor
from practitioner import Practitioner
from topology.cs_endpoint import ClientEndpoint


class Worker(Executor):
    def __init__(
        self,
        task_id: int,
        worker_id: int,
        endpoint: ClientEndpoint,
        practitioner: Practitioner,
        **kwargs: Any,
    ) -> None:
        name = f"worker {worker_id}"
        if task_id is not None:
            name = f"worker {worker_id} of {task_id}"
        super().__init__(name=name, **kwargs)
        self.__worker_id = worker_id
        self.__practitioner: Practitioner = practitioner
        self._endpoint = endpoint
        self._round_num = 0
        self._force_stop = False

    @property
    def worker_id(self):
        return self.__worker_id

    @cached_property
    def trainer(self) -> Trainer:
        return self.new_trainer()

    def new_trainer(self) -> Trainer:
        return self.__practitioner.create_trainer(self.config)

    def _offload_from_device(self) -> None:
        self.trainer.offload_from_device()
        if self.trainer.has_hook_obj("keep_model_hook"):
            self.trainer.get_hook("keep_model_hook").clear()
        # self.trainer.model_util.clear_parameters()
        # for phase in MachineLearningPhase:
        #     inferencer = self.trainer.get_cached_inferencer(phase=phase)
        #     if inferencer is not None:
        # #         inferencer.model_util.clear_parameters()
        # print(total_size(self))

    def _before_training(self) -> None:
        pass

    def _stopped(self) -> bool:
        return self._round_num > self.config.round or self._force_stop

    def start(self, **kwargs: Any) -> None:
        first_training: bool = True
        self._round_num = 1
        self._force_stop = False
        while not self._stopped():
            # in case worker changes round number
            with self._get_execution_context():
                if first_training:
                    self._before_training()
                    first_training = False
                    # in case worker changes round number
                    if self._stopped():
                        break
                    self.trainer.set_device(
                        device=self._get_device(
                            lock_callback=lambda: self.trainer.append_named_hook(
                                ExecutorHookPoint.AFTER_BATCH,
                                "release_device_lock",
                                self._release_device_lock,
                            )
                        )
                    )
                else:
                    self.trainer.disable_hook("logger")
                self.trainer.set_visualizer_prefix(prefix=f"round: {self._round_num},")
                self.trainer.train(
                    keep_best_model=True,
                    batch_loss_log_times=None if self.config.log_batch_loss else 0,
                    **kwargs,
                )
                self._round_num += 1
        get_logger().debug("finish worker %s", self.worker_id)
        get_logger().debug("close endpoint")
        self._endpoint.close()
