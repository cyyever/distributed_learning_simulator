import gevent

from .worker import Worker


class Client(Worker):
    def send_data_to_server(self, data):
        self._endpoint.send(data)

    def _get_result_from_server(self) -> dict:
        self._release_semaphore()
        while True:
            self._acquire_semaphore()
            if self._endpoint.has_data():
                break
            self._release_semaphore()
            gevent.sleep(1)
        return self._endpoint.get()
