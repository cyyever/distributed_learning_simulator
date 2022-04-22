import os
import threading

import gevent
import gevent.lock
from cyy_naive_lib.log import get_logger
from executor import Executer
from topology.endpoint import Endpoint


class Server(Executer):
    def __init__(self, config, endpoint: Endpoint):
        super().__init__(config=config, endpoint=endpoint, name="server")
        self.round_number = 0
        self._end_server: bool = False

    def start(self):
        while not self._end_server:
            for worker_id in range(self._endpoint._topology.worker_num):
                while True:
                    self._acquire_semaphore()
                    has_request = self._endpoint.has_data(worker_id)
                    if not has_request:
                        self._release_semaphore()
                        gevent.sleep(1)
                        continue
                    self._process_worker_data(
                        worker_id, self._endpoint.get(worker_id=worker_id)
                    )
                    self._release_semaphore()
                    break

    def _process_worker_data(self, worker_id, data):
        raise NotImplementedError()


    @property
    def worker_number(self):
        return self.config.worker_number

    def send_result(self, data):
        selected_workers = self._select_workers()
        get_logger().debug("choose workers %s", selected_workers)
        for worker_id in range(self.worker_number):
            result = None
            if worker_id in selected_workers:
                result = data
            self._endpoint.send(data=result, worker_id=worker_id)

    def _select_workers(self) -> set:
        return set(range(self.worker_number))

    def _acquire_semaphore(self):
        super()._acquire_semaphore()
        threading.current_thread().name = "server"
