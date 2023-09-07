import copy
import multiprocessing
import os
import threading
from typing import Any, Callable

import gevent.local
import gevent.lock
import torch
from cyy_torch_toolbox.device import get_device

from config import DistributedTrainingConfig
from topology.endpoint import Endpoint


class ExecutorContext:
    semaphore = gevent.lock.BoundedSemaphore(value=1)
    local_data = gevent.local.local()

    def __init__(self, name: str) -> None:
        self.__name: str = name

    def acquire(self) -> None:
        self.semaphore.acquire()
        multiprocessing.current_process().name = self.__name
        threading.current_thread().name = self.__name
        ExecutorContext.local_data.ctx = self

    def __enter__(self) -> None:
        self.acquire()

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.release()

    def release(self) -> None:
        multiprocessing.current_process().name = "unknown executor"
        threading.current_thread().name = "unknown executor"
        self.semaphore.release()


class Executor:
    __thread_data = threading.local()

    def __init__(
        self,
        config: DistributedTrainingConfig,
        name: str,
        device_lock: threading.RLock,
    ) -> None:
        self.config: DistributedTrainingConfig = copy.deepcopy(config)
        self.__used_device_memory = None
        self.__name = name
        self.__device_lock: threading.RLock = device_lock
        self.__hold_device_lock: bool = False
        assert self.config.save_dir is not None
        self.config.save_dir = os.path.abspath(
            os.path.join(self.config.save_dir, self.__name.replace(" ", "_"))
        )

    def _get_device(self, lock_callback: None | Callable = None) -> torch.device:
        if not hasattr(self.__thread_data, "device"):
            if not self.__hold_device_lock:
                self.__device_lock.acquire()
                self.__hold_device_lock = True
                if lock_callback is not None:
                    lock_callback()
            self.__thread_data.device = get_device(
                max_needed_bytes=self.__used_device_memory
            )
            if "cuda" in self.__thread_data.device.type.lower():
                torch.cuda.set_device(self.__thread_data.device)
        return self.__thread_data.device

    def _get_execution_context(self) -> ExecutorContext:
        return ExecutorContext(name=self.__name)

    def _release_device_lock(self, **kwargs: Any) -> None:
        if self.__hold_device_lock:
            if "cuda" in self.__thread_data.device.type.lower():
                stats = torch.cuda.memory_stats(device=self.__thread_data.device)
                if stats:
                    self.__used_device_memory = stats["allocated_bytes.all.peak"]
            self.__device_lock.release()
            self.__hold_device_lock = False

    def start(self) -> None:
        raise NotImplementedError()
