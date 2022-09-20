import copy
import multiprocessing
import os
import threading

import gevent
import gevent.lock
import torch
from cyy_torch_toolbox.device import get_device


class Executor:
    semaphore = gevent.lock.BoundedSemaphore(value=1)
    __thread_data = threading.local()
    __hold_device_lock = False

    def __init__(self, config, endpoint, name, **kwargs):
        self.config = copy.deepcopy(config)
        self._endpoint = endpoint
        self.__used_cuda_memory = None
        self.__hold_semaphore: bool = False
        self._name = name
        self.__device_lock = kwargs.get("device_lock", None)

    def _get_device(self, lock_callback=None):
        if not hasattr(self.__thread_data, "device"):
            if not self.__hold_device_lock:
                self.__device_lock.acquire()
                self.__hold_device_lock = True
                if lock_callback is not None:
                    lock_callback()
            self.__thread_data.device = get_device(
                use_cuda_only=True, max_needed_bytes=self.__used_cuda_memory
            )
            torch.cuda.set_device(self.__thread_data.device)
        return self.__thread_data.device

    def _acquire_semaphore(self):
        if not self.__hold_semaphore:
            self.semaphore.acquire()
            multiprocessing.current_process().name = self._name
            threading.current_thread().name = self._name
            self.__hold_semaphore = True

    def _release_semaphore(self):
        if self.__hold_semaphore:
            multiprocessing.current_process().name = "unknown executor"
            threading.current_thread().name = "unknown executor"
            self.__hold_semaphore = False
            self.semaphore.release()

    def _release_device_lock(self, **kwargs):
        if self.__hold_device_lock:
            if hasattr(self.__thread_data, "device"):
                stats = torch.cuda.memory_stats(device=self.__thread_data.device)
                if stats:
                    self.__used_cuda_memory = stats["allocated_bytes.all.peak"]
            self.__device_lock.release()
            self.__hold_device_lock = False

    def start(self):
        raise NotImplementedError()

    @property
    def save_dir(self):
        save_dir = os.path.join(self.config.save_dir, self._name.replace(" ", "_"))
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
