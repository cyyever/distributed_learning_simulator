import os
import sys
from pathlib import Path

from distributed_learning_simulation.config import import_dependencies

import_dependencies(dataset_type=os.environ.get("CYY_DATASET_TYPE"))

sys.path.insert(0, os.path.abspath("."))
import method  # noqa: F401

if __name__ == "__main__":
    from distributed_learning_simulation import load_config, train

    config_path = Path(__file__).parent / "conf"
    config = load_config(
        config_path=config_path,
        global_conf_path=config_path / "global.yaml",
        import_libs=False,
    )
    train(config=config, single_task=True)
