import multiprocessing
import os
import sys
import uuid

import gevent

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
from cyy_naive_lib.data_structure.process_initialization import \
    get_process_data
from cyy_naive_lib.log import get_logger, set_file_handler
from cyy_naive_lib.system_info import get_operating_system
from cyy_torch_toolbox.data_structure.torch_process_pool import \
    TorchProcessPool

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from algorithm_factory import get_worker_config
from config import DistributedTrainingConfig


def start_executors(
    task_id: int | None,
    worker_configs: list[dict],
    server_config: None | dict = None,
) -> dict:

    device_lock = get_process_data()["device_lock"]
    topology = get_process_data()["topology"]
    workers: list = []

    for worker_config in worker_configs:
        practitioner = worker_config.pop("practitioner")
        config = worker_config.pop("config")
        trainer = practitioner.create_trainer(config=config)
        workers.append(
            worker_config["constructor"](
                extra_kwargs={
                    "task_id": task_id,
                    "device_lock": device_lock,
                    "trainer": trainer,
                },
                extra_endpoint_kwargs={
                    "topology": topology,
                },
            )
        )
    if server_config is not None:
        get_logger().debug("run server with other workers in the same process")
        server_constructor = server_config.pop("constructor")
        workers.append(
            server_constructor(
                extra_kwargs={
                    "task_id": task_id,
                    "device_lock": device_lock,
                },
                extra_endpoint_kwargs={
                    "topology": topology,
                },
            )
        )

    get_logger().debug("run workers")
    gevent.joinall([gevent.spawn(worker.start) for worker in workers], raise_error=True)
    get_logger().debug("stop process")

    res: dict = {}
    for worker in workers:
        if not hasattr(worker, "worker_id"):
            # server
            server = worker
            if hasattr(server, "sv_algorithm"):
                res["sv"] = server.sv_algorithm.shapley_values
            res |= server.performance_stat[server.round_number]
            continue
    return res


tasks: dict = {}


def train(
    config: DistributedTrainingConfig,
    non_blocking: bool = False,
    practitioner_ids: None | set = None,
) -> int | None:
    if hasattr(os, "sysconf"):
        name = "SC_OPEN_MAX"
        value = os.sysconf(name)
        if isinstance(value, int) and value <= 1024:
            raise RuntimeError(
                f"Your open file limit {value} is too small, the training will open lots of files."
            )
    config.apply_global_config()
    set_file_handler(config.log_file)
    worker_config = get_worker_config(config, practitioner_ids=practitioner_ids)
    topology = worker_config.pop("topology")
    device_lock = multiprocessing.Manager().RLock()
    task_id: int | None = uuid.uuid4().int
    if not non_blocking:
        task_id = None
    process_pool: TorchProcessPool = TorchProcessPool(
        method="spawn" if get_operating_system() != "freebsd" else "fork",
        initargs=[{"fun_kwargs": {"device_lock": device_lock, "topology": topology}}],
    )
    for process_idx, worker_configs in worker_config["worker"].items():
        server_config = None
        if process_idx == 0:
            server_config = worker_config.get("server", None)
        process_pool.exec(
            start_executors,
            task_id=task_id,
            worker_configs=worker_configs,
            server_config=server_config,
        )
    if not non_blocking:
        process_pool.shutdown()
        return None
    tasks[task_id] = {
        "process_pool": process_pool,
        "practitioner_ids": practitioner_ids,
        "config": config,
    }
    return task_id


def get_training_result(task_id: int, timeout: None | float = None) -> None | dict:
    task = tasks[task_id]
    process_pool = task["process_pool"]
    if not process_pool.wait(timeout=timeout):
        return None
    tasks.pop(task_id)
    results = process_pool.wait_results()
    process_pool.shutdown()
    tmp_stats: dict = {}
    for result in results:
        tmp_stats |= result
    stats: dict = {}
    practitioner_ids = task["practitioner_ids"]
    config = task["config"]
    if practitioner_ids is not None:
        for k, v in tmp_stats.items():
            if k != "sv":
                stats[k] = v
                continue
            sv_dict: dict = {}
            for round, tmp_sv_dict in v.items():
                sv_dict[round] = {}
                for practitioner_id, worker_id in zip(
                    sorted(practitioner_ids), range(config.worker_number)
                ):
                    sv_dict[round][practitioner_id] = tmp_sv_dict[worker_id]
            stats[k] = sv_dict
    else:
        stats = tmp_stats
    return stats
