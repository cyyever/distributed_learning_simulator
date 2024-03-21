import importlib
import os

from .common_import import *  # noqa: F401

for entry in os.scandir(os.path.dirname(os.path.abspath(__file__))):
    if not entry.is_dir():
        continue
    if entry.name == "__pycache__":
        continue
    if entry.name.startswith("."):
        continue
    try:
        importlib.import_module(f".{entry.name}", "method")
        continue
    except BaseException:
        pass
    try:
        importlib.import_module(
            f".{entry.name}", "distributed_learning_simulator.method"
        )
    except BaseException:
        pass
