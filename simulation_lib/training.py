import copy
import multiprocessing
import os

# we use these env variables to save memory in large-scale training
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["USE_THREAD_DATALOADER"] = "1"
import uuid

import gevent
from cyy_naive_lib.data_structure.process_initialization import \
    get_process_data
from cyy_naive_lib.log import add_file_handler, get_logger
from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_toolbox.data_structure.torch_process_pool import \
    TorchProcessPool

from .algorithm_factory import get_worker_config
from .config import DistributedTrainingConfig


def start_server(task_id: int | None, server_config: dict) -> dict:
    device_lock = get_process_data()["device_lock"]
    topology = get_process_data()["topology"]
    get_logger().debug("task_id %s topology id %d", task_id, id(topology))

    server = server_config["constructor"](
        extra_kwargs={
            "task_id": task_id,
            "device_lock": device_lock,
        },
        extra_endpoint_kwargs={
            "topology": topology,
        },
    )

    server.start()
    get_logger().info("stop server")

    res: dict = {}
    if hasattr(server.algorithm, "shapley_values"):
        res["sv"] = server.algorithm.shapley_values
    res |= {"performance": server.performance_stat}
    return res


def start_workers(
    task_id: int | None,
    worker_configs: list[dict],
) -> None:
    device_lock = get_process_data()["device_lock"]
    log_lock = get_process_data()["log_lock"]
    topology = get_process_data()["topology"]
    workers: list = []
    assert worker_configs

    for worker_config in worker_configs:
        workers.append(
            worker_config["constructor"](
                extra_kwargs={
                    "task_id": task_id,
                    "device_lock": device_lock,
                    "log_lock": log_lock,
                },
                extra_endpoint_kwargs={
                    "topology": topology,
                },
            )
        )
    get_logger().debug(
        "run workers %s in the same process for task %s",
        [worker.worker_id for worker in workers],
        task_id,
    )

    gevent.joinall([gevent.spawn(worker.start) for worker in workers], raise_error=True)
    get_logger().info("stop workers")


tasks: dict = {}
task_results: dict = {}


def train(
    config: DistributedTrainingConfig,
    practitioners: None | set = None,
) -> int | None:
    # we need to deepcopy config for concurrent training
    config = copy.deepcopy(config)
    practitioners = copy.deepcopy(practitioners)
    config.reset_session()
    config.apply_global_config()
    timer = TimeCounter()
    if hasattr(os, "sysconf"):
        name = "SC_OPEN_MAX"
        value = os.sysconf(name)
        if isinstance(value, int) and value <= 1024:
            raise RuntimeError(
                f"Your open file limit {value} is too small, the training will open lots of files."
            )
    task_id = None
    if practitioners is None:
        add_file_handler(config.log_file)
    else:
        task_id = uuid.uuid4().int + os.getpid()
    worker_config = get_worker_config(config, practitioners=practitioners)
    topology = worker_config.pop("topology")
    device_lock = multiprocessing.Manager().RLock()
    log_lock = multiprocessing.Manager().Semaphore()
    assert topology.worker_num == config.worker_number
    process_pool: TorchProcessPool = TorchProcessPool(
        initargs=[
            {
                "fun_kwargs": {
                    "device_lock": device_lock,
                    "log_lock": log_lock,
                    "topology": topology,
                }
            }
        ],
    )
    for worker_configs in worker_config["worker"]:
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
    if practitioners is not None:
        tasks[task_id] = {
            "process_pool": process_pool,
            "practitioner_ids": {practitioner.id for practitioner in practitioners},
            "config": config,
        }
        task_results[task_id] = {}
        return task_id
    process_pool.wait_results(timeout=None)
    process_pool.shutdown(wait=True)
    get_logger().info("training use %s seconds", timer.elapsed_milliseconds() / 1000)
    return None


def get_training_result(task_id: int, timeout: None | float = None) -> None | dict:
    task = tasks[task_id]
    process_pool = task["process_pool"]
    results, not_done = process_pool.wait_results(timeout=timeout)
    for result in results.values():
        if result is not None:
            task_results[task_id] |= result
    if not_done:
        return None
    process_pool.shutdown()
    tasks.pop(task_id)
    get_logger().info("finish task %s", task_id)
    stats: dict = {}
    practitioner_ids = task["practitioner_ids"]
    config = task["config"]
    assert practitioner_ids is not None
    for k, v in task_results[task_id].items():
        if k != "sv":
            stats[k] = v
            continue
        sv_dict: dict = {}
        for round_number, tmp_sv_dict in v.items():
            sv_dict[round_number] = {}
            for practitioner_id, worker_id in zip(
                sorted(practitioner_ids), range(config.worker_number)
            ):
                sv_dict[round_number][practitioner_id] = tmp_sv_dict[worker_id]
        stats[k] = sv_dict
    task_results.pop(task_id)
    return stats
