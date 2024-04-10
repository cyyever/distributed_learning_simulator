import networkx
import torch_geometric.data
import torch_geometric.utils
from distributed_learning_simulation import (
    AggregationAlgorithm, CompositeAggregationAlgorithm,
    GraphNodeEmbeddingPassingAlgorithm, GraphTopologyAlgorithm, Message,
    PersonalizedFedAVGAlgorithm)


class GraphDistanceAlgorithm(GraphTopologyAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self._client_distance: dict = {}
        self._training_node_to_client: dict = {}
        self._aggregation_algorithm: AggregationAlgorithm | None = None

    def set_aggregation_algorithm(self, algorithm: AggregationAlgorithm) -> None:
        self._aggregation_algorithm = algorithm

    def process_worker_data(self, worker_id: int, worker_data: Message | None) -> bool:
        if (
            worker_data is None
            or "compute_client_distance" not in worker_data.other_data
        ):
            return super().process_worker_data(
                worker_id=worker_id, worker_data=worker_data
            )

        worker_data.other_data.pop("compute_client_distance")
        edge_index = worker_data.other_data.pop("edge_index")
        if not self._client_distance:
            assert self._training_node_indices
            for client, nodes in self._training_node_indices.items():
                for node in nodes:
                    self._training_node_to_client[node] = client
            G = torch_geometric.utils.to_networkx(
                data=torch_geometric.data.Data(edge_index=edge_index)
            )
            for source, target_dict in networkx.all_pairs_shortest_path_length(G):
                for target, distance in target_dict.items():
                    if (
                        source in self._training_node_to_client
                        and target in self._training_node_to_client
                    ):
                        source_client = self._training_node_to_client[source]
                        target_client = self._training_node_to_client[target]
                        if source_client == target_client:
                            continue
                        pair = (source_client, target_client)
                        if source_client > target_client:
                            pair = (target_client, source_client)
                        if pair not in self._client_distance:
                            self._client_distance[pair] = []
                        self._client_distance[pair].append(distance)
            assert self._client_distance
        assert not worker_data.other_data
        return True

    def aggregate_worker_data(self) -> Message:
        if not self._client_distance:
            return super().aggregate_worker_data()
        assert self._client_distance
        worker_weights: dict = {}
        for (a, b), v in self._client_distance.items():
            if len(v) > 1:
                mean_distance = sum(v) / len(v)
            else:
                mean_distance = v
            if a not in worker_weights:
                worker_weights[a] = {}
            if b not in worker_weights:
                worker_weights[b] = {}
            worker_weights[a][b] = mean_distance
            worker_weights[b][a] = mean_distance
        assert isinstance(self._aggregation_algorithm, PersonalizedFedAVGAlgorithm)
        self._aggregation_algorithm.set_worker_weights(worker_weights)
        self._client_distance.clear()
        return Message(in_round=True)


class GraphFedAlgorithm(CompositeAggregationAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        distance_algorithm = GraphDistanceAlgorithm()
        self.append_algorithm(distance_algorithm)
        self.append_algorithm(GraphNodeEmbeddingPassingAlgorithm())
        avg_algorithm = PersonalizedFedAVGAlgorithm()
        distance_algorithm.set_aggregation_algorithm(avg_algorithm)
        self.append_algorithm(avg_algorithm)
