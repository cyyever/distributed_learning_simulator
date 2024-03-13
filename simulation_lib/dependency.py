import importlib


def import_dependencies() -> dict:
    result = {}
    for dependency in ("cyy_torch_graph", "cyy_torch_text", "cyy_torch_vision"):
        try:
            importlib.import_module(dependency)
            result[dependency] = True
        except BaseException:
            pass
    return result


import_results = import_dependencies()
