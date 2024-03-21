from .composite_aggregation_algorithm import CompositeAggregationAlgorithm
from .fed_avg_algorithm import FedAVGAlgorithm
from .graph_embedding_algorithm import GraphNodeEmbeddingPassingAlgorithm
from .graph_topology_algorithm import GraphTopologyAlgorithm


class GraphAlgorithm(CompositeAggregationAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.append_algorithm(GraphTopologyAlgorithm())
        self.append_algorithm(GraphNodeEmbeddingPassingAlgorithm())
        self.append_algorithm(FedAVGAlgorithm())
