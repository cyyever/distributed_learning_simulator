from ..common_import import AggregationServer


class ShapleyValueServer(AggregationServer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.need_init_performance = True
