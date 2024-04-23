import torch_geometric.utils
from distributed_learning_simulation import GraphWorker, Message


class GraphFedWorker(GraphWorker):
    def _before_training(self) -> None:
        self._remove_cross_edge = False
        super()._before_training()
        edge_index = self.edge_index.clone()
        edge_index = torch_geometric.utils.coalesce(
            edge_index[:, self.cross_client_edge_mask]
        )
        sent_data = Message(
            other_data={
                "compute_client_distance": True,
                "edge_index": edge_index,
            },
            in_round=True,
        )
        if not self._share_feature:
            self._clear_cross_client_edges()
        self.send_data_to_server(sent_data)
        self._get_data_from_server()
