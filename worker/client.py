import gevent
from executor import ExecutorContext

from .worker import Worker


class Client(Worker):
    def send_data_to_server(self, data):
        self._endpoint.send(data)

    def _get_result_from_server(self) -> dict:
        assert (
            hasattr(ExecutorContext.thread_data, "ctx")
            and ExecutorContext.thread_data.ctx is not None
        )
        ExecutorContext.thread_data.ctx.release()
        while True:
            ExecutorContext.thread_data.ctx.acquire()
            if self._endpoint.has_data():
                break
            ExecutorContext.thread_data.ctx.release()
            gevent.sleep(1)
        return self._endpoint.get()
