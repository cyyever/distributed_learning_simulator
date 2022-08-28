import os
import sys
import threading

sys.path.insert(0, os.path.abspath("."))


import gevent
from cyy_naive_lib.log import get_logger

from util.data_spliting import DataSplitter

local_data = threading.local()


def create_worker(
    worker_id,
    worker_constructor,
    data_splitter: DataSplitter,
    config,
    device_lock,
    topology,
    endpoint_cls,
):
    trainer = config.create_trainer()
    trainer.set_save_dir(os.path.join(trainer.save_dir, "worker_" + str(worker_id)))
    data_splitter.split(trainer, worker_id)
    return worker_constructor(
        config=config,
        trainer=trainer,
        worker_id=worker_id,
        device_lock=device_lock,
        endpoint=endpoint_cls(topology=topology),
    )


def process_initializer(device_lock, topology):
    global local_data
    local_data.device_lock = device_lock
    local_data.topology = topology


def run_workers(worker_configs, server_config=None):
    global local_data
    device_lock = local_data.device_lock
    topology = local_data.topology

    workers = []

    for worker_config in worker_configs:
        workers.append(
            create_worker(**worker_config, device_lock=device_lock, topology=topology)
        )
    if server_config is not None:
        get_logger().debug("run server with other workers in the same process")
        endpoint_cls = server_config.pop("server_endpoint_cls")
        server_constructor = server_config.pop("server_constructor")
        workers.append(
            server_constructor(
                device_lock=device_lock, endpoint=endpoint_cls(topology=topology)
            )
        )

    get_logger().debug("run workers")
    tasks = [gevent.spawn(worker.start) for worker in workers]
    gevent.joinall(tasks, raise_error=True)
    get_logger().debug("stop process")
