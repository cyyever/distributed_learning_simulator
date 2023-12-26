import os
import sys

sys.path.insert(0, os.path.abspath("."))


from simulation_lib.config import global_config, load_config
from simulation_lib.training import train

if __name__ == "__main__":
    load_config()
    global_config.apply_global_config()
    train(config=global_config)
