from typing import Any, Iterable

from .central_topology import CentralTopology
from .endpoint import Endpoint


class ServerEndpoint(Endpoint):
    _topology: CentralTopology

    @property
    def worker_num(self):
        return self._topology.worker_num

    def has_data(self, worker_id: int) -> bool:
        return self._topology.worker_has_data(worker_id=worker_id)

    def get(self, worker_id):
        return self._topology.get_from_worker(worker_id=worker_id)

    def send(self, worker_id: int, data: Any) -> None:
        self._topology.send_to_worker(worker_id=worker_id, data=data)

    def broadcast(self, data: Any, worker_ids: None | Iterable = None) -> None:
        all_worker_ids = set(range(self.worker_num))
        if worker_ids is None:
            worker_ids = all_worker_ids
        else:
            worker_ids = set(worker_ids).intersection(all_worker_ids)
        for worker_id in worker_ids:
            self.send(worker_id=worker_id, data=data)

    def wait_close(self):
        self._topology.wait_close()


class ClientEndpoint(Endpoint):
    _topology: CentralTopology

    def __init__(self, topology: CentralTopology, worker_id: int):
        super().__init__(topology=topology)
        self.__worker_id: int = worker_id

    def get(self):
        return self._topology.get_from_server(self.__worker_id)

    def has_data(self) -> bool:
        return self._topology.server_has_data(self.__worker_id)

    def send(self, data):
        self._topology.send_to_server(worker_id=self.__worker_id, data=data)

    def close(self):
        self._topology.close()
