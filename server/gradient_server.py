from server.aggregation_server import AggregationServer


class GradientServer(AggregationServer):
    __end: bool = False

    def _process_worker_data(self, worker_id, data) -> None:
        if "end_training" in data:
            self.__end = True
            return
        super()._process_worker_data(worker_id, data)

    def _stopped(self) -> bool:
        return self.__end
