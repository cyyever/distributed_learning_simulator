from functools import cached_property
from itertools import chain
from sys import getsizeof
from typing import Any

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import ExecutorHookPoint
from cyy_torch_toolbox.trainer import Trainer
from executor import Executor
from practitioner import Practitioner


def total_size(o, handlers={}):
    """Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """

    def dict_handler(d):
        return chain.from_iterable(d.items())

    all_handlers = {
        tuple: iter,
        list: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)
        if isinstance(o, float | int | bool | str):
            return s
        if isinstance(o, torch.Tensor):
            return o.element_size() * o.nelement()
        for attr in dir(o):
            if attr.startswith("__"):
                continue
            if hasattr(o, attr):
                value = getattr(o, attr)
                if hasattr(value, "__call__"):
                    continue
                # print("attr is", attr, type(value))
                s += sizeof(value)

        return s

    return sizeof(o)


class Worker(Executor):
    def __init__(
        self,
        task_id: int,
        worker_id: int,
        practitioner: Practitioner,
        **kwargs: Any,
    ) -> None:
        name = f"worker {worker_id}"
        if task_id is not None:
            name = f"worker {worker_id} of {task_id}"
        super().__init__(name=name, **kwargs)
        self.__worker_id = worker_id
        self.__practitioner: Practitioner = practitioner
        self._round_num = 0
        self._force_stop = False

    @property
    def worker_id(self):
        return self.__worker_id

    @cached_property
    def trainer(self) -> Trainer:
        return self.__practitioner.create_trainer(self.config)

    def _offload_from_device(self) -> None:
        self.trainer.offload_from_device()
        self.trainer.get_hook("keep_model_hook").clear()
        # self.trainer.model_util.clear_parameters()
        # for phase in MachineLearningPhase:
        #     inferencer = self.trainer.get_cached_inferencer(phase=phase)
        #     if inferencer is not None:
        # #         inferencer.model_util.clear_parameters()
        # if not hasattr(self, "_keep_optimizer"):
        #     self.trainer.remove_optimizer()
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
                        self._get_device(
                            lock_callback=lambda: self.trainer.append_named_hook(
                                ExecutorHookPoint.AFTER_BATCH,
                                "release_device_lock",
                                self._release_device_lock,
                            )
                        )
                    )
                else:
                    self.trainer.disable_hook("logger")
                self.trainer.set_visualizer_prefix(f"round: {self._round_num},")
                self.trainer.train(
                    keep_best_model=True,
                    batch_loss_log_times=None if self.config.log_batch_loss else 0,
                    **kwargs,
                )
                self._round_num += 1
        get_logger().debug("finish worker %s", self.worker_id)
        get_logger().debug("close endpoint")
        self._endpoint.close()
