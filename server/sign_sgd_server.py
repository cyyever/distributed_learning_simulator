""" signSGD: Compressed Optimisation for Non-Convex Problems https://arxiv.org/abs/1802.04434 """

from algorithm.sign_sgd_algorithm import SignSGDAlgorithm

from .aggregation_server import AggregationServer


class SignSGDServer(AggregationServer):
    __end = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, algorithm=SignSGDAlgorithm(), **kwargs)

    def _process_worker_data(self, worker_id, data):
        if "end_training" in data:
            self.__end = True
            return
        super()._process_worker_data(worker_id, data)

    def _stopped(self) -> bool:
        return self.__end
