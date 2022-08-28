import random

from .fed_avg_server import FedAVGServer


class FedAVGRandomSubsetServer(FedAVGServer):
    def _select_workers(self) -> set:
        return set(
            random.sample(
                list(range(self.worker_number)),
                k=self.config.algorithm_kwargs["random_client_number"],
            )
        )
