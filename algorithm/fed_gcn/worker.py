from worker.graph_worker import GraphWorker


class FedGCNWorker(GraphWorker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._share_feature = True
