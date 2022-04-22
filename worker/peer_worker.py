import gevent

from .worker import Worker


class PeerWorker(Worker):
    peer_data = None

    def broadcast(self, data):
        self._endpoint.broadcast(data)

    def gather(self) -> dict:
        res: dict = {}
        for worker_id in self._endpoint.foreach_peer():
            while True:
                has_result = self._endpoint.has_data(peer_id=worker_id)
                if has_result:
                    res[worker_id] = self._endpoint.get(peer_id=worker_id)
                    break
                self._release_semaphore()
                gevent.sleep(1)
                self._acquire_semaphore()
        return res
