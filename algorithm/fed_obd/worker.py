from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import ExecutorHookPoint

from worker.fed_avg_worker import FedAVGWorker

from .obd_algorithm import OpportunisticBlockDropoutAlgorithm
from .phase import Phase


class FedOBDWorker(FedAVGWorker, OpportunisticBlockDropoutAlgorithm):
    __phase = Phase.STAGE_ONE
    __end_training = False

    def __init__(self, *args, **kwargs):
        FedAVGWorker.__init__(self, *args, **kwargs)
        OpportunisticBlockDropoutAlgorithm.__init__(
            self, dropout_rate=self.config.algorithm_kwargs["dropout_rate"]
        )
        self._endpoint.dequant_server_data = True
        self._keep_optimizer = True
        self._send_parameter_diff = False

    def _load_result_from_server(self, result):
        if "phase_two" in result:
            result.pop("phase_two")
            self.__phase = Phase.STAGE_TWO
            get_logger().warning("switch to phase 2")
            self._reuse_learning_rate = True
            self._send_parameter_diff = True
            self._choose_model_by_validation = False
            self.trainer.hyper_parameter.epoch = self.config.algorithm_kwargs[
                "second_phase_epoch"
            ]
            self.config.round = self._round_num + 1
            get_logger().warning(
                "change epoch to %s", self.trainer.hyper_parameter.epoch
            )
            self._aggregation_time = ExecutorHookPoint.AFTER_EPOCH
            self._register_aggregation()

        super()._load_result_from_server(result=result)

    def _aggregation(self, sent_data, **kwargs):
        if self.__phase == Phase.STAGE_TWO:
            executor = kwargs["executor"]
            if kwargs["epoch"] == executor.hyper_parameter.epoch:
                sent_data["final_aggregation"] = True
                self.__end_training = True
        super()._aggregation(sent_data=sent_data, **kwargs)

    def _stopped(self) -> bool:
        return self.__end_training

    def _get_sent_data(self):
        data = super()._get_sent_data()
        if self.__phase == Phase.STAGE_ONE:
            block_parameter = self.get_block_parameter(
                parameter_dict=data.pop("parameter"),
                model_util=self.trainer.model_util,
                model_cache=self._model_cache,
            )
            data["parameter_diff"] = self._model_cache.get_parameter_diff(
                block_parameter
            )
            return data

        data["in_round_data"] = True
        data["check_acc"] = True
        get_logger().warning("phase 2 keys %s", data.keys())
        return data
