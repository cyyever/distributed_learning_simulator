import os
import sys
from contextlib import redirect_stdout

from cyy_naive_lib.log import (
    StreamToLogger,
    replace_default_logger,
)
from distributed_learning_simulation import import_dependencies, load_config, train

sys.path.insert(0, os.path.abspath("."))
import method  # noqa: F401

import_dependencies()
if __name__ == "__main__":
    with redirect_stdout(StreamToLogger()):
        replace_default_logger()
        config_path = os.path.join(os.path.dirname(__file__), "conf")
        config = load_config(
            config_path=config_path,
            global_conf_path=os.path.join(config_path, "global.yaml"),
        )
        train(config=config, single_task=True)
