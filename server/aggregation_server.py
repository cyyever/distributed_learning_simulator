import torch
from cyy_naive_lib.log import get_logger

from .server import Server


class AggregationServer(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_number = 0
        self.__worker_data = {}

    def _process_worker_data(self, worker_id, data):
        assert 0 <= worker_id < self.worker_number
        self.__worker_data[worker_id] = data
        if len(self.__worker_data) == self.worker_number:
            self.round_number += 1
            if self.round_number == self.config.round:
                self._end_server = True
            self.send_result(
                self._aggregate_worker_data(
                    self.round_number,
                    {k: v for k, v in self.__worker_data.items() if v is not None},
                )
            )
            self.__worker_data.clear()
            torch.cuda.empty_cache()
        else:
            get_logger().debug(
                "we have %s committed, and we need %s workers,skip",
                set(self.__worker_data.keys()),
                self.worker_number,
            )
