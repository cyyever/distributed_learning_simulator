from distributed_learning_simulation import Message
from distributed_learning_simulation.worker.graph_worker import GraphWorker
from torch import Tensor


class FedClusterGCNWorker(GraphWorker):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._share_feature = True
        self._remove_cross_edge = False

    def _get_cross_device_edges(self) -> set:
        assert self._other_training_node_indices
        edge_index: Tensor = self.edge_index
        edge_mask: Tensor = self.training_node_mask[edge_index[0]]
        edge_index = edge_index[:, edge_mask]
        worker_boundary = set()
        for a, b in edge_index.transpose(0, 1).numpy():
            if b in self._other_training_node_indices:
                worker_boundary.add((a, b))
        assert worker_boundary
        return worker_boundary

    def _determine_topology(self) -> None:
        super()._determine_topology()
        sent_data = Message(
            other_data={
                "cross_device_edges": self._get_cross_device_edges(),
                "cluster_number": self.config.algorithm_kwargs["cluster_number"],
            },
            in_round=True,
        )
        self._send_data_to_server(sent_data)
        res = self._get_data_from_server()
        assert isinstance(res, Message)
        self._other_training_node_indices = res.other_data[
            "other_training_node_indices"
        ][self.worker_id]
        assert self._other_training_node_indices
