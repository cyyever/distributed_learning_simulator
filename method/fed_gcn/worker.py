from distributed_learning_simulation import GraphWorker


class FedGCNWorker(GraphWorker):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._share_feature = True
