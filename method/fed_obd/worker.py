from typing import Any

from cyy_naive_lib.log import log_debug, log_warning
from cyy_torch_toolbox import ExecutorHookPoint, ModelUtil
from distributed_learning_simulation import (
    AggregationWorker,
    Message,
    ParameterMessage,
    QuantClientEndpoint,
)
from distributed_learning_simulation.context import ClientEndpointInCoroutine

from .obd_algorithm import OpportunisticBlockDropoutAlgorithmMixin
from .phase import Phase


class FedOBDWorker(AggregationWorker, OpportunisticBlockDropoutAlgorithmMixin):
    __phase = Phase.STAGE_ONE

    def __init__(self, *args, **kwargs):
        AggregationWorker.__init__(self, *args, **kwargs)
        OpportunisticBlockDropoutAlgorithmMixin.__init__(self)
        assert isinstance(
            self._endpoint, QuantClientEndpoint | ClientEndpointInCoroutine
        )
        self._endpoint.dequant_server_data()
        self._send_parameter_diff = False
        self._keep_model_cache = True

    def _load_result_from_server(self, result: Message) -> None:
        if "phase_two" in result.other_data:
            assert isinstance(result, ParameterMessage)
            # result.other_data.pop("phase_two")
            self.__phase = Phase.STAGE_TWO
            log_warning("switch to phase 2")
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

    def _get_model_util(self) -> ModelUtil:
        return self.trainer.model_util

    def _aggregation(self, sent_data: Message, **kwargs: Any) -> None:
        if self.__phase == Phase.STAGE_TWO:
            executor = kwargs["executor"]
            if kwargs["epoch"] == executor.hyper_parameter.epoch:
                sent_data.end_training = True
                self._force_stop = True
                log_debug("end training")
        super()._aggregation(sent_data=sent_data, **kwargs)

    def _stopped(self) -> bool:
        return self._force_stop

    def _get_sent_data(self):
        assert self._model_cache is not None
        data = super()._get_sent_data()
        if self.__phase == Phase.STAGE_ONE:
            assert isinstance(data, ParameterMessage)
            block_parameter = self.get_block_parameter(
                parameter=data.parameter,
            )
            data.parameter = self._model_cache.get_parameter_diff(block_parameter)
            return data

        data.in_round = True
        log_warning("phase 2 keys %s", data.other_data.keys())
        return data
