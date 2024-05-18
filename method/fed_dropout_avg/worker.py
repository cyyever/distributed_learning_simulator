""" FedDropoutAvg: Generalizable federated learning for histopathology image classification (https://arxiv.org/pdf/2111.13230.pdf) """

import torch
from cyy_naive_lib.log import get_logger
from distributed_learning_simulation import AggregationWorker, ParameterMessage


class FedDropoutAvgWorker(AggregationWorker):

    def _get_sent_data(self) -> ParameterMessage:
        dropout_rate: float = self.config.algorithm_kwargs["dropout_rate"]
        if self.hold_log_lock:
            get_logger().info("use dropout_rate %s", dropout_rate)
        self._send_parameter_diff = False
        sent_data = super()._get_sent_data()
        assert isinstance(sent_data, ParameterMessage)
        parameter = sent_data.parameter
        total_num: float = 0
        send_num: float = 0
        for k, v in parameter.items():
            weight = torch.bernoulli(torch.full_like(v, 1 - dropout_rate))
            parameter[k] = v * weight
            total_num += parameter[k].numel()
            send_num += torch.count_nonzero(parameter[k]).item()
        get_logger().error("send_num %s", send_num)
        get_logger().error("total_num %s", total_num)
        return sent_data
