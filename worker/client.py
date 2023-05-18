from typing import Any

import gevent
from executor import ExecutorContext

from .worker import Worker


class Client(Worker):
    def send_data_to_server(self, data: Any) -> None:
        self._endpoint.send(data)

    def _get_result_from_server(self) -> dict:
        ExecutorContext.local_data.ctx.release()
        self._release_device_lock()
        while True:
            ExecutorContext.local_data.ctx.acquire()
            if self._endpoint.has_data():
                break
            ExecutorContext.local_data.ctx.release()
            gevent.sleep(0.1)
        return self._endpoint.get()
