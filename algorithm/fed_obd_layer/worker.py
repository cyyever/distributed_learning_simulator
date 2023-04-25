from algorithm.fed_obd.phase import Phase
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import ExecutorHookPoint

from worker.fed_avg_worker import FedAVGWorker

from .obd_layer_algorithm import OpportunisticLayerDropoutAlgorithm


class FedOBDLayerWorker(FedAVGWorker, OpportunisticLayerDropoutAlgorithm):
    __phase = Phase.STAGE_ONE
    __end_training = False

    def __init__(self, *args, **kwargs):
        FedAVGWorker.__init__(self, *args, **kwargs)
        OpportunisticLayerDropoutAlgorithm.__init__(
            self,
            dropout_rate=self.config.algorithm_kwargs["dropout_rate"],
        )
        self._endpoint.dequant_server_data = True
        self._send_parameter_diff = False
        self._find_blocks()

    def _load_result_from_server(self, result):
        if "phase_two" in result:
            result.pop("phase_two")
            self.__phase = Phase.STAGE_TWO
            get_logger().warning("switch to phase 2")
            self._reuse_learning_rate = True
            self.trainer.remove_optimizer()
            self.trainer.hyper_parameter.set_epoch(
                self.config.algorithm_kwargs["second_phase_epoch"]
            )
            self.config.round = self._round_num + 1
            get_logger().warning(
                "change epoch to %s", self.trainer.hyper_parameter.epoch
            )
            get_logger().warning(
                "change lr to %s",
                self.trainer.hyper_parameter.get_learning_rate(self.trainer),
            )
            self._aggregation_time = ExecutorHookPoint.AFTER_EPOCH
            self._register_aggregation()

        super()._load_result_from_server(result=result)

    def _should_aggregate(self, **kwargs):
        if self.__phase != Phase.STAGE_TWO:
            return True
        model_executor = kwargs["model_executor"]
        if kwargs["epoch"] == model_executor.hyper_parameter.epoch:
            self.__end_training = True
        return True

    def _stopped(self) -> bool:
        return self.__end_training

    def _get_sent_data(self):
        data = super()._get_sent_data()
        if self.__phase == Phase.STAGE_ONE:
            data["parameter"] = self.get_block_parameter(data["parameter"])
            return super()._get_sent_data()

        self._choose_model_by_validation = False

        data["in_round_data"] = True
        data["check_acc"] = True
        if self.__end_training:
            data["final_aggregation"] = True
        return data
