import functools

from .central_topology import CentralTopology
from .endpoint import Endpoint
from .quantized_endpoint import QuantizedEndpoint


class ServerEndpoint(Endpoint):
    _topology: CentralTopology

    def set_function(self, fun):
        self._topology.set_server_function(fun)

    def has_data(self, worker_id: int) -> bool:
        return self._topology.worker_has_data(worker_id=worker_id)

    def get(self, worker_id):
        return self._topology.get_from_worker(worker_id=worker_id)

    def send(self, data, worker_id):
        self._topology.send_to_worker(data, worker_id)

    def broadcast(self, data):
        for worker_id in range(self._topology.worker_num):
            self.send(data, worker_id)

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
        self._topology.send_to_server(self.__worker_id, data)

    def close(self):
        self._topology.close()


class QuantizedClientEndpoint(ClientEndpoint, QuantizedEndpoint):
    def send(self, data):
        super().send(data=self.quant(data))


class QuantizedServerEndpoint(ServerEndpoint, QuantizedEndpoint):
    def set_function(self, fun):
        self._topology.set_server_function(
            functools.partial(self.dequan_and_process, fun=fun)
        )

    def dequan_and_process(self, data, fun):
        return fun(self.dequant(data))
