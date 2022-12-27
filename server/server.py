import random

import gevent
import gevent.lock
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.model_executor import ModelExecutor
from executor import Executor


class Server(Executor):
    def __init__(self, task_id, **kwargs):
        name = "server"
        if task_id is not None:
            name = f"server of {task_id}"
        super().__init__(**kwargs, name=name)
        self.__tester: None | ModelExecutor = None

    @property
    def tester(self) -> ModelExecutor:
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
            self.__tester.disable_logger()
        return self.__tester

    def get_metric(self, parameter_dict: dict) -> dict:
        if "parameter" in parameter_dict:
            parameter_dict = parameter_dict["parameter"]
        self.tester.model_util.load_parameter_dict(parameter_dict)
        self.tester.model_util.disable_running_stats()
        self.tester.set_device(self._get_device())
        self.tester.inference(epoch=1)
        metric = {
            "acc": self.tester.performance_metric.get_accuracy(1).item(),
            "loss": self.tester.performance_metric.get_loss(1).item(),
        }
        self._release_device_lock()
        self.tester.offload_from_gpu()
        return metric

    def start(self):
        while not self._stopped():
            for worker_id in range(self._endpoint.worker_num):
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
        get_logger().warning("end server")

    def _process_worker_data(self, worker_id, data):
        raise NotImplementedError()

    @property
    def worker_number(self):
        return self.config.worker_number

    def send_result(self, data):
        selected_workers = self._select_workers()
        get_logger().debug("choose workers %s", selected_workers)
        if selected_workers:
            self._endpoint.broadcast(data=data, worker_ids=selected_workers)
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
