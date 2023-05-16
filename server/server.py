import copy
import random

import gevent
import gevent.lock
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.executor import Executor as CentralizedExecutor
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from executor import Executor


class Server(Executor):
    def __init__(self, task_id: int, **kwargs: dict) -> None:
        name: str = "server"
        if task_id is not None:
            name = f"server of {task_id}"
        super().__init__(**kwargs, name=name)
        self.__tester: None | CentralizedExecutor = None

    @property
    def tester(self) -> CentralizedExecutor:
        if self.__tester is None:
            self.__tester = self.config.create_inferencer(
                phase=MachineLearningPhase.Test
            )
            self.__tester.dataset_collection.remove_dataset(
                phase=MachineLearningPhase.Training
            )
            self.__tester.dataset_collection.remove_dataset(
                phase=MachineLearningPhase.Validation
            )
            self.__tester.disable_hook("logger")
        return self.__tester

    def get_metric(
        self, parameter_dict: dict, keep_performance_logger: bool = True
    ) -> dict:
        if "parameter" in parameter_dict:
            parameter_dict = parameter_dict["parameter"]
        self.tester.model_util.load_parameter_dict(parameter_dict)
        self.tester.model_util.disable_running_stats()
        self.tester.set_save_dir(self.save_dir)
        self.tester.set_device(self._get_device())
        if keep_performance_logger:
            self.__tester.enable_hook("performance_metric_logger")
        else:
            self.__tester.disable_hook("performance_metric_logger")
        self.tester.inference(epoch=1)
        metric = {
            "acc": self.tester.performance_metric.get_accuracy(1).item(),
            "loss": self.tester.performance_metric.get_loss(1).item(),
        }
        self._release_device_lock()
        self.tester.offload_from_gpu()
        return metric

    def start(self):
        worker_set = set()
        while not self._stopped():
            if not worker_set:
                worker_set = set(range(self._endpoint.worker_num))
            with self._get_context():
                for worker_id in copy.copy(worker_set):
                    has_data: bool = self._endpoint.has_data(worker_id)
                    if has_data:
                        self._process_worker_data(
                            worker_id, self._endpoint.get(worker_id=worker_id)
                        )
                        get_logger().debug("get result from %s", worker_id)
                        worker_set.remove(worker_id)
            if worker_set and not self._stopped():
                get_logger().debug("wait result")
                gevent.sleep(1)

        with self._get_context():
            get_logger().warning("end server")
            self._server_exit()

    def _server_exit(self) -> None:
        pass

    def _process_worker_data(self, worker_id, data):
        raise NotImplementedError()

    @property
    def worker_number(self):
        return self.config.worker_number

    def send_result(self, result):
        if "worker_result" in result:
            for worker_id, data in result["worker_result"].items():
                self._endpoint.send(data, worker_id)
            return

        selected_workers = self._select_workers()
        get_logger().debug("choose workers %s", selected_workers)
        if selected_workers:
            self._endpoint.broadcast(data=result, worker_ids=selected_workers)
        unselected_workers = set(range(self.worker_number)) - selected_workers
        if unselected_workers:
            self._endpoint.broadcast(data=None, worker_ids=unselected_workers)

    def _select_workers(self) -> set:
        if "random_client_number" in self.config.algorithm_kwargs:
            return set(
                random.sample(
                    list(range(self.worker_number)),
                    k=self.config.algorithm_kwargs["random_client_number"],
                )
            )
        return set(range(self.worker_number))

    def _stopped(self) -> bool:
        raise NotImplementedError()
