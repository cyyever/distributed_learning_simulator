from algorithm.fed_obd.phase import Phase
from algorithm.random_dropout_algorithm import RandomDropoutAlgorithm
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import ExecutorHookPoint

from worker.fed_avg_worker import FedAVGWorker


class FedOBDRandomDropoutWorker(FedAVGWorker, RandomDropoutAlgorithm):
    __phase = Phase.STAGE_ONE
    __end_training = False

    def __init__(self, *args, **kwargs):
        FedAVGWorker.__init__(self, *args, **kwargs)
        RandomDropoutAlgorithm.__init__(
            self, dropout_rate=self.config.algorithm_kwargs["dropout_rate"]
        )
        self._endpoint.dequant_server_data = True

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
        # distributed first time from server
        if self._model_cache.get_parameter_dict is None:
            super()._load_result_from_server(result=result)
            return

        if "parameter_diff" in result:
            parameter = {}
            for k, v in result["parameter_diff"].items():
                parameter[k] = self._model_cache.get_parameter_dict[k] + v
            result["parameter"] = parameter
        super()._load_result_from_server(result=result)

    def _should_aggregate(self, **kwargs):
        if self.__phase != Phase.STAGE_TWO:
            return True
        model_executor = kwargs["model_executor"]
        if kwargs["epoch"] == model_executor.hyper_parameter.epoch:
            self.__end_training = True
        return True

    def _get_sent_data(self):
        data = super()._get_sent_data()
        if self.__phase == Phase.STAGE_ONE:
            data["parameter"] = self.drop_parameter(data["parameter"])
            return super()._get_sent_data()

        self._choose_model_by_validation = False

        data["in_round_data"] = True
        data["check_acc"] = True
        if self.__end_training:
            data["final_aggregation"] = True
        return data
