import os
import sys

import cyy_huggingface_toolbox  # noqa: F401
import cyy_torch_text  # noqa: F401
import cyy_torch_vision  # noqa: F401
from cyy_naive_lib.log import redirect_stdout_to_logger
from distributed_learning_simulation import load_config, train

sys.path.insert(0, os.path.abspath("."))
import method  # noqa: F401

if __name__ == "__main__":
    with redirect_stdout_to_logger():
        config_path = os.path.join(os.path.dirname(__file__), "conf")
        config = load_config(
            config_path=config_path,
            global_conf_path=os.path.join(config_path, "global.yaml"),
        )
        train(config=config, single_task=True)
