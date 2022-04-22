from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue

from .topology import Topology


class CentralTopology(Topology):
    def __init__(self, worker_num):
        self.worker_num = worker_num

    def get_from_server(self, worker_id):
        raise NotImplementedError()

    def set_server_function(self, fun):
        raise NotImplementedError()

    def get_from_worker(self, worker_id):
        raise NotImplementedError()

    def server_has_data(self, worker_id: int) -> bool:
        raise NotImplementedError()

    def worker_has_data(self, worker_id: int) -> bool:
        raise NotImplementedError()

    def send_to_server(self, worker_id, data):
        raise NotImplementedError()

    def send_to_worker(self, data, worker_id):
        raise NotImplementedError()

    def wait_close(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class ProcessCentralTopology(CentralTopology):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__queue = TorchProcessTaskQueue(
            worker_num=0, use_manager=True, move_data_in_cpu=True
        )
        for worker_id in range(self.worker_num):
            self.__queue.add_result_queue(f"result_{worker_id}")
            self.__queue.add_result_queue(f"request_{worker_id}")
        self.__global_lock = None

    def create_global_lock(self):
        self.__global_lock = self.__queue.get_manager().RLock()

    @property
    def global_lock(self):
        return self.__global_lock

    def get_from_worker(self, worker_id):
        assert 0 <= worker_id < self.worker_num
        return self.__queue.get_result(queue_name=f"request_{worker_id}")

    def get_from_server(self, worker_id):
        assert 0 <= worker_id < self.worker_num
        return self.__queue.get_result(queue_name=f"result_{worker_id}")

    def set_server_function(self, fun):
        self.__queue.set_worker_fun(fun)

    def server_has_data(self, worker_id: int) -> bool:
        assert 0 <= worker_id < self.worker_num
        return self.__queue.has_result(queue_name=f"result_{worker_id}")

    def worker_has_data(self, worker_id: int) -> bool:
        assert 0 <= worker_id < self.worker_num
        return self.__queue.has_result(queue_name=f"request_{worker_id}")

    def send_to_server(self, worker_id, data):
        assert 0 <= worker_id < self.worker_num
        self.__queue.put_result(data, queue_name=f"request_{worker_id}")

    def send_to_worker(self, data, worker_id):
        self.__queue.put_result(result=data, queue_name=f"result_{worker_id}")

    def wait_close(self):
        self.__queue.join()

    def close(self):
        self.__queue.stop()
