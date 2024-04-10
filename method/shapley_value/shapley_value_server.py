from distributed_learning_simulation import AggregationServer


class ShapleyValueServer(AggregationServer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._need_init_performance = True
