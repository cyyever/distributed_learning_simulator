import copy

from algorithm.fed_avg_algorithm import FedAVGAlgorithm
from cyy_torch_toolbox.device import get_cpu_device, put_data_to_device

from .aggregation_server import AggregationServer


class FedAVGServer(AggregationServer, FedAVGAlgorithm):
    def start(self):
        self._acquire_semaphore()
        self._prev_model = put_data_to_device(
            copy.deepcopy(self.tester.model_util.get_parameter_dict()),
            device=get_cpu_device(),
        )
        assert self.prev_model is not None
        # save GPU memory
        self.tester.offload_from_gpu()
        if not self.config.no_distribute_init_parameters:
            self.send_result(data=self.prev_model)
        self._release_semaphore()
        super().start()
