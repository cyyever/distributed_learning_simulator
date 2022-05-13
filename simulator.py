import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import gevent
from cyy_naive_lib.log import get_logger, set_file_handler
from cyy_torch_toolbox.data_structure.torch_process_pool import \
    TorchProcessPool
from cyy_torch_toolbox.dataset import sub_dataset
from cyy_torch_toolbox.dataset_transform.transforms import replace_target
from cyy_torch_toolbox.ml_type import MachineLearningPhase, TransformType

from config import get_config
from factory import get_worker_config


def create_worker(
    worker_id,
    worker_constructor,
    training_dataset_indices,
    config,
    noise_percent,
):
    trainer = config.create_trainer()
    trainer.set_save_dir(os.path.join(trainer.save_dir, "worker_" + str(worker_id)))
    get_logger().debug("use indices with first ten %s", training_dataset_indices[:10])
    trainer.dataset_collection.transform_dataset(
        MachineLearningPhase.Training,
        lambda training_dataset, _: sub_dataset(
            training_dataset, training_dataset_indices
        ),
    )
    if noise_percent is not None and noise_percent != 0:
        get_logger().warning("use noise_percent %s", noise_percent)

        label_map = trainer.dataset_collection.get_dataset_util(
            phase=MachineLearningPhase.Training
        ).randomize_subset_label(noise_percent)
        trainer.dataset_collection.append_transform(
            transform=replace_target(label_map),
            key=TransformType.Target,
            phases=[MachineLearningPhase.Training],
        )
    return worker_constructor(
        config=config,
        trainer=trainer,
        worker_id=worker_id,
    )


def run_workers(worker_configs, server_config=None):
    workers = []

    for worker_config in worker_configs:
        workers.append(create_worker(**worker_config))
    if server_config is not None:
        get_logger().debug("run server with other workers in the same process")
        workers.append(server_config())

    get_logger().info("run workers")
    tasks = [gevent.spawn(worker.start) for worker in workers]
    gevent.joinall(tasks, raise_error=True)
    get_logger().info("stop process")


if __name__ == "__main__":
    if hasattr(os, "sysconf"):
        name = "SC_OPEN_MAX"
        value = os.sysconf(name)
        if isinstance(value, int) and value <= 1024:
            raise RuntimeError(
                f"Your open file limit {value} is too small, this training uses lots of open files."
            )
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    os.environ["USE_THREAD_DATALOADER"] = "1"
    parser = argparse.ArgumentParser()
    config = get_config(parser=parser)
    config.apply_global_config()
    set_file_handler(config.log_file)
    get_logger().warning("arguments are %s", parser.parse_args())
    result = get_worker_config(config)

    process_pool = TorchProcessPool()
    for process_idx, worker_configs in result["worker_map"].items():
        server_config = None
        if process_idx == 0 and "server_constructor" in result:
            server_config = result["server_constructor"]
        process_pool.exec(
            run_workers, worker_configs=worker_configs, server_config=server_config
        )
    process_pool.stop()
