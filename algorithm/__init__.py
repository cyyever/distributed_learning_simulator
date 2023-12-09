import cyy_torch_text  # noqa: F401
import cyy_torch_vision  # noqa: F401

from .fed_avg import *  # noqa: F401
from .fed_dropout_avg import *  # noqa: F401

try:
    import cyy_torch_graph  # noqa: F401

    from .fed_gcn import *  # noqa: F401
    from .fed_gnn import *  # noqa: F401
except BaseException:
    pass

from .afd import *  # noqa: F401
from .fed_obd import *  # noqa: F401
from .fed_paq import *  # noqa: F401
from .qsgd import *  # noqa: F401
from .shapley_value import *  # noqa: F401
from .sign_sgd import *  # noqa: F401


def register_algorithms():
    pass
