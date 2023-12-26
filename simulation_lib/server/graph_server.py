from ..algorithm.graph_algorithm import GraphNodeEmbeddingPassingAlgorithm
from .aggregation_server import AggregationServer


class GraphNodeServer(AggregationServer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(GraphNodeEmbeddingPassingAlgorithm(), *args, **kwargs)
