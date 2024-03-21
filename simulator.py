import os
import sys

sys.path.insert(0, os.path.abspath("."))


from simulation_lib.config import global_config, load_config
from simulation_lib.training import train
import method

if __name__ == "__main__":
    load_config()
    train(config=global_config)
