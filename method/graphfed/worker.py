from cyy_torch_toolbox import MachineLearningPhase
from distributed_learning_simulation import GraphWorker, Message


class GraphFedWorker(GraphWorker):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._share_feature = True

    def _before_training(self) -> None:
        super()._before_training()
        sent_data = Message(
            other_data={
                "compute_client_distance": True,
                "edge_index": self.get_dataset_util(
                    phase=MachineLearningPhase.Training
                ).get_edge_index(graph_index=0),
            },
            in_round=True,
        )
        self.send_data_to_server(sent_data)
        self._get_data_from_server()
