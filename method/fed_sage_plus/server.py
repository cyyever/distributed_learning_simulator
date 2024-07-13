from cyy_torch_toolbox import Inferencer
from distributed_learning_simulation import AggregationServer

from .evaluator import replace_evaluator


class FedSagePlusServer(AggregationServer):
    def get_tester(self) -> Inferencer:
        tester = super().get_tester()
        replace_evaluator(tester)
        return tester
