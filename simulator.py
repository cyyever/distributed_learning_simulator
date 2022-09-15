import os
import sys

sys.path.insert(0, os.path.abspath("."))

from config import global_config, load_config
from training import train

if __name__ == "__main__":
    load_config()
    config = global_config
    train(config)
