import multiprocessing
import os
import sys

sys.path.insert(0, os.path.abspath("."))


from cyy_naive_lib.log import set_file_handler
from cyy_torch_toolbox.data_structure.torch_process_pool import \
    TorchProcessPool

from config import global_config, load_config
from factory import get_worker_config
from process import process_initializer, run_workers

if __name__ == "__main__":
    if hasattr(os, "sysconf"):
        name = "SC_OPEN_MAX"
        value = os.sysconf(name)
        if isinstance(value, int) and value <= 1024:
            raise RuntimeError(
                f"Your open file limit {value} is too small, this training uses lots of open files."
            )
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    load_config()
    config = global_config
    config.apply_global_config()
    set_file_handler(config.log_file)
    worker_config = get_worker_config(config)
    topology = worker_config.pop("topology")

    device_lock = multiprocessing.Manager().RLock()

    process_pool = TorchProcessPool(
        initializer=process_initializer, initargs=(device_lock, topology)
    )
    for process_idx, worker_configs in worker_config["worker_map"].items():
        server_config = None
        if process_idx == 0:
            server_config = worker_config.get("server_config", None)
        process_pool.exec(
            run_workers, worker_configs=worker_configs, server_config=server_config
        )
    process_pool.stop()
