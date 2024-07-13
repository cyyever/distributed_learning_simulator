import importlib
import os

from distributed_learning_simulation.dependency import \
    import_results  # noqa: F401

for entry in os.scandir(os.path.dirname(os.path.abspath(__file__))):
    if not entry.is_dir():
        continue
    if entry.name == "__pycache__":
        continue
    if entry.name.startswith("."):
        continue
    importlib.import_module(f".{entry.name}", "method")
