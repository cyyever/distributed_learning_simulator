import functools

import networkx
import torch
import torch_geometric.data
import torch_geometric.utils
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.concurrency import TorchProcessTaskQueue
from distributed_learning_simulation import (
    AggregationAlgorithm, CompositeAggregationAlgorithm,
    GraphNodeEmbeddingPassingAlgorithm, GraphTopologyAlgorithm, Message,
    PersonalizedFedAVGAlgorithm)


def compute_shortest_paths(training_node_to_client, task, **kwargs) -> dict:
    client_distance: dict = {}
    G = torch_geometric.utils.to_networkx(
        data=torch_geometric.data.Data(edge_index=task)
    )
    for source, target_dict in networkx.all_pairs_shortest_path_length(G):
        for target, distance in target_dict.items():
            if source in training_node_to_client and target in training_node_to_client:
                source_client = training_node_to_client[source]
                target_client = training_node_to_client[target]
                if source_client == target_client:
                    continue
                pair = (source_client, target_client)
                if source_client > target_client:
                    pair = (target_client, source_client)
                if pair not in client_distance:
                    client_distance[pair] = []
                client_distance[pair].append(distance)
    return client_distance


class GraphDistanceAlgorithm(GraphTopologyAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self._client_distance: dict = {}
        self._training_node_to_client: dict = {}
        self._edge_indices: list = []
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
        self._edge_indices.append(edge_index)
        assert not worker_data.other_data
        return True

    def aggregate_worker_data(self) -> Message:
        if not self._edge_indices and not self._client_distance:
            return super().aggregate_worker_data()

        if not self._client_distance:
            assert self._training_node_indices
            for client, nodes in self._training_node_indices.items():
                for node in nodes:
                    self._training_node_to_client[node] = client
            assert self._edge_indices
            edge_index = torch_geometric.utils.to_undirected(
                torch_geometric.utils.coalesce(torch.concat(self._edge_indices, dim=1))
            )
            self._edge_indices = []

            clients = sorted(self._training_node_indices.keys())
            max_index = torch.max(edge_index.view(-1)).item()
            assert isinstance(max_index, int)
            queue = TorchProcessTaskQueue(worker_num=20)
            queue.disable_logger()
            get_logger().info("start queue")
            queue.start(
                worker_fun=functools.partial(
                    compute_shortest_paths, self._training_node_to_client
                )
            )
            get_logger().info("end queue")
            cnt = 0
            for idx, client in enumerate(clients):
                for client2 in clients[idx + 1:]:
                    cnt += 1
                    training_node_indices = self._training_node_indices[client].union(
                        self._training_node_indices[client2]
                    )
                    node_mask = torch_geometric.utils.index_to_mask(
                        torch.tensor(list(training_node_indices)),
                        size=max(max_index, *training_node_indices) + 1,
                    )
                    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                    pair_edge_index = edge_index[:, edge_mask]
                    get_logger().info("shape is %s", pair_edge_index.shape)
                    queue.add_task(pair_edge_index)
                    while queue.has_data():
                        cnt -= 1
                        res = queue.get_data()
                        get_logger().debug("get data ")
                        assert res is not None
                        res = res[0]
                        self._client_distance |= res
            for _ in range(cnt):
                res = queue.get_data()
                get_logger().debug("get data ")
                assert res is not None
                res = res[0]
                self._client_distance |= res
            queue.stop()
            assert self._client_distance

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
