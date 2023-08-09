from typing import Any

from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue

from .topology import Topology


class CentralTopology(Topology):
    def __init__(self, worker_num):
        self.worker_num = worker_num

    def get_from_server(self, worker_id: int):
        raise NotImplementedError()

    def get_from_worker(self, worker_id: int):
        raise NotImplementedError()

    def server_has_data(self, worker_id: int) -> bool:
        raise NotImplementedError()

    def worker_has_data(self, worker_id: int) -> bool:
        raise NotImplementedError()

    def send_to_server(self, worker_id: int, data):
        raise NotImplementedError()

    def send_to_worker(self, worker_id: int, data):
        raise NotImplementedError()

    def wait_close(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class ProcessCentralTopology(CentralTopology):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__queue = TorchProcessTaskQueue(
            worker_num=0,
            use_manager=False,
        )
        for worker_id in range(self.worker_num):
            self.__queue.add_queue(f"result_{worker_id}")
            self.__queue.add_queue(f"request_{worker_id}")

    def get_from_worker(self, worker_id: int) -> Any:
        assert 0 <= worker_id < self.worker_num
        return self.__queue.get_data(queue_name=f"request_{worker_id}")[0]

    def get_from_server(self, worker_id: int) -> Any:
        assert 0 <= worker_id < self.worker_num
        return self.__queue.get_data(queue_name=f"result_{worker_id}")[0]

    def server_has_data(self, worker_id: int) -> bool:
        assert 0 <= worker_id < self.worker_num
        return self.__queue.has_data(queue_name=f"result_{worker_id}")

    def worker_has_data(self, worker_id: int) -> bool:
        assert 0 <= worker_id < self.worker_num
        return self.__queue.has_data(queue_name=f"request_{worker_id}")

    def send_to_server(self, worker_id: int, data: Any) -> None:
        assert 0 <= worker_id < self.worker_num
        self.__queue.put_data(data=data, queue_name=f"request_{worker_id}")

    def send_to_worker(self, worker_id: int, data: Any) -> None:
        self.__queue.put_data(data=data, queue_name=f"result_{worker_id}")

    def wait_close(self) -> None:
        self.__queue.join()

    def close(self) -> None:
        self.__queue.stop()
