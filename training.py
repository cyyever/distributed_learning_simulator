import multiprocessing
import os

# we use these env variables to save memory in large-scale training
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["USE_THREAD_DATALOADER"] = "1"
import sys
import uuid

import gevent
from cyy_naive_lib.data_structure.process_initialization import \
    get_process_data
from cyy_naive_lib.log import add_file_handler, get_logger
from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_toolbox.data_structure.torch_process_pool import \
    TorchProcessPool

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from algorithm_factory import get_worker_config
from config import DistributedTrainingConfig


def start_server(task_id: int, server_config: dict) -> dict:
    device_lock = get_process_data()["device_lock"]
    topology = get_process_data()["topology"]

    server = server_config["constructor"](
        extra_kwargs={
            "task_id": task_id,
            "device_lock": device_lock,
        },
        extra_endpoint_kwargs={
            "topology": topology,
        },
    )

    get_logger().debug("run workers")
    gevent.joinall([gevent.spawn(server.start)], raise_error=True)
    get_logger().debug("stop process")

    res: dict = {}
    if hasattr(server.algorithm, "shapley_values"):
        res["sv"] = server.algorithm.shapley_values
    res |= {"performance": server.performance_stat}
    return res


def start_workers(
    task_id: int,
    worker_configs: list[dict],
) -> None:
    device_lock = get_process_data()["device_lock"]
    topology = get_process_data()["topology"]
    workers: list = []
    assert worker_configs

    for worker_config in worker_configs:
        workers.append(
            worker_config["constructor"](
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


tasks: dict = {}


def train(
    config: DistributedTrainingConfig,
    non_blocking: bool = False,
    practitioner_ids: None | set = None,
) -> int | None:
    timer = TimeCounter()
    if hasattr(os, "sysconf"):
        name = "SC_OPEN_MAX"
        value = os.sysconf(name)
        if isinstance(value, int) and value <= 1024:
            raise RuntimeError(
                f"Your open file limit {value} is too small, the training will open lots of files."
            )
    config.apply_global_config()
    add_file_handler(config.log_file)
    worker_config = get_worker_config(config, practitioner_ids=practitioner_ids)
    topology = worker_config.pop("topology")
    device_lock = multiprocessing.Manager().RLock()
    task_id = uuid.uuid4().int
    process_pool: TorchProcessPool = TorchProcessPool(
        initargs=[{"fun_kwargs": {"device_lock": device_lock, "topology": topology}}],
    )
    for worker_configs in worker_config["worker"].values():
        process_pool.submit(
            start_workers, task_id=task_id, worker_configs=worker_configs
        )
    server_config = worker_config.get("server", None)
    if server_config is not None:
        process_pool.submit(
            start_server,
            task_id=task_id,
            server_config=server_config,
        )
    if non_blocking:
        tasks[task_id] = {
            "process_pool": process_pool,
            "practitioner_ids": practitioner_ids,
            "config": config,
        }
        return task_id
    process_pool.wait_results(timeout=None)
    process_pool.shutdown(wait=True)
    get_logger().info("training use %s seconds", timer.elapsed_milliseconds() / 1000)


def get_training_result(task_id: int, timeout: None | float = None) -> None | dict:
    task = tasks[task_id]
    process_pool = task["process_pool"]
    results, not_done = process_pool.wait_results(timeout=timeout)
    process_pool.shutdown()
    if not_done:
        return None
    tasks.pop(task_id)
    tmp_stats: dict = {}
    for result in results.values():
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
