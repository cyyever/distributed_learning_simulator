from .fed_avg import *
from .fed_dropout_avg import *

try:
    from .fed_gcn import *
    from .fed_gnn import *
    from .fed_aas import *
except BaseException:
    pass
from .afd import *
from .fed_obd import *
from .fed_paq import *
from .qsgd import *
from .shapley_value import *
from .sign_sgd import *


def register_algorithms():
    pass
