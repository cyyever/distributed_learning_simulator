import os
import sys

sys.path.insert(0, os.path.abspath("."))

import hydra
from distributed_learning_simulation.config import DistributedTrainingConfig
from distributed_learning_simulation.config import load_config as __load_config
from distributed_learning_simulation.training import train

import method  # noqa: F401

global_config: DistributedTrainingConfig = DistributedTrainingConfig()


@hydra.main(config_path="./conf", version_base=None)
def load_config(conf) -> None:
    global global_config
    global_config = __load_config(
        conf, os.path.join(os.path.dirname(__file__), "conf", "global.yaml")
    )


if __name__ == "__main__":
    load_config()
    train(config=global_config)
