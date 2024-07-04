import json
import os

import torch
import torch_geometric.utils
from cyy_torch_graph.dataset.partition import METIS
from distributed_learning_simulation.algorithm.fed_avg_algorithm import \
    FedAVGAlgorithm
from distributed_learning_simulation.algorithm.graph_algorithm import (
    CompositeAggregationAlgorithm, GraphNodeEmbeddingPassingAlgorithm,
    GraphTopologyAlgorithm)
from distributed_learning_simulation.message import Message


class ClusterAlgorithm(GraphTopologyAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.__worker_cross_device_edges: dict[int, torch.Tensor] = {}
        self.__cluster_number = 0

    def process_worker_data(self, worker_id: int, worker_data: Message | None) -> bool:
        if worker_data is not None and "cross_device_edges" in worker_data.other_data:
            self.__worker_cross_device_edges[worker_id] = worker_data.other_data.pop(
                "cross_device_edges"
            )
            self.__cluster_number = worker_data.other_data.pop("cluster_number")
            assert not worker_data.other_data
            return True
        return super().process_worker_data(worker_id=worker_id, worker_data=worker_data)

    def aggregate_worker_data(self) -> Message:
        if not self.__worker_cross_device_edges:
            return super().aggregate_worker_data()
        virtual_edge_weights = {}
        node_to_client = {}
        for worker_id, node_indices in self._training_node_indices.items():
            for node_index in node_indices:
                node_to_client[node_index] = worker_id
        for edges in self.__worker_cross_device_edges.values():
            for source, dest in edges:
                source_client = node_to_client[source]
                dest_client = node_to_client[dest]
                if source_client >= dest_client:
                    continue
                pair = (source_client, dest_client)
                if pair not in virtual_edge_weights:
                    virtual_edge_weights[pair] = 1
                else:
                    virtual_edge_weights[pair] += 1
        assert virtual_edge_weights
        edge_weights: dict = {}
        pair_list = []
        for pair, weight in virtual_edge_weights.items():
            pair_list.append(list(pair))
            edge_weights[pair] = weight

        edge_index = torch_geometric.utils.to_undirected(
            torch.tensor(pair_list, dtype=torch.long).transpose(0, 1)
        )
        clusters = METIS(
            edge_index=edge_index,
            num_nodes=len(self._training_node_indices),
            num_parts=self.__cluster_number,
            edge_weights=edge_weights,
        )
        cluster_allocation = dict(enumerate(clusters.tolist()))
        cluster_path = os.path.join(
            self.config.save_dir,
            "cluster.json",
        )
        with open(cluster_path, "wt", encoding="utf8") as f:
            json.dump(cluster_allocation, f)
        other_training_node_indices: dict[int, set] = {}
        for worker_id, cluster_no in cluster_allocation.items():
            other_training_node_indices[worker_id] = set()
            for other_worker_id, other_cluster_no in cluster_allocation.items():
                if other_cluster_no == cluster_no and other_worker_id != worker_id:
                    other_training_node_indices[worker_id] = (
                        other_training_node_indices[worker_id]
                        | self._training_node_indices[other_worker_id]
                    )
            assert other_training_node_indices[worker_id]

        return Message(
            in_round=True,
            other_data={"other_training_node_indices": other_training_node_indices},
        )


class FedClusterGCNAlgorithm(CompositeAggregationAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.append_algorithm(ClusterAlgorithm())
        self.append_algorithm(GraphNodeEmbeddingPassingAlgorithm())
        self.append_algorithm(FedAVGAlgorithm())
