from algorithm.graph_algorithm import GraphNodeEmbeddingPassingAlgorithm

from server.fed_avg_server import FedAVGServer


class GraphNodeServer(FedAVGServer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args, algorithm=GraphNodeEmbeddingPassingAlgorithm(), **kwargs
        )
