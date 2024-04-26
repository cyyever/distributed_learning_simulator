from typing import Any

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import ExecutorHookPoint
from distributed_learning_simulation import (AggregationWorker, Message,
                                             ParameterMessage,
                                             QuantClientEndpoint)

from .obd_algorithm import OpportunisticBlockDropoutAlgorithmMixin
from .phase import Phase


class FedOBDWorker(AggregationWorker, OpportunisticBlockDropoutAlgorithmMixin):
    __phase = Phase.STAGE_ONE

    def __init__(self, *args, **kwargs):
        AggregationWorker.__init__(self, *args, **kwargs)
        OpportunisticBlockDropoutAlgorithmMixin.__init__(self)
        assert isinstance(self._endpoint, QuantClientEndpoint)
        self._endpoint.dequant_server_data = True
        self._send_parameter_diff = False
        self._keep_model_cache = True

    def _load_result_from_server(self, result: Message) -> None:
        if "phase_two" in result.other_data:
            assert isinstance(result, ParameterMessage)
            # result.other_data.pop("phase_two")
            self.__phase = Phase.STAGE_TWO
            get_logger().warning("switch to phase 2")
            self._reuse_learning_rate = True
            self._send_parameter_diff = True
            self.disable_choosing_model_by_validation()
            self.trainer.hyper_parameter.epoch = self.config.algorithm_kwargs[
                "second_phase_epoch"
            ]
            self.config.round = self._round_index + 1
            self._aggregation_time = ExecutorHookPoint.AFTER_EPOCH
            self._register_aggregation()

        super()._load_result_from_server(result=result)

    def _aggregation(self, sent_data: Message, **kwargs: Any) -> None:
        if self.__phase == Phase.STAGE_TWO:
            executor = kwargs["executor"]
            if kwargs["epoch"] == executor.hyper_parameter.epoch:
                sent_data.end_training = True
                self._force_stop = True
                get_logger().debug("end training")
        super()._aggregation(sent_data=sent_data, **kwargs)

    def _stopped(self) -> bool:
        return self._force_stop

    def _get_sent_data(self):
        assert self._model_cache is not None
        data = super()._get_sent_data()
        if self.__phase == Phase.STAGE_ONE:
            assert isinstance(data, ParameterMessage)
            block_parameter = self.get_block_parameter(
                parameter_dict=data.parameter,
            )
            data.parameter = self._model_cache.get_parameter_diff(block_parameter)
            return data

        data.in_round = True
        get_logger().warning("phase 2 keys %s", data.other_data.keys())
        return data
