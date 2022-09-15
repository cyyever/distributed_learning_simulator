import random

from algorithm.fed_obd_phase import Phase
from cyy_naive_lib.log import get_logger

from .fed_avg_server import FedAVGServer


class FedOBDServer(FedAVGServer):
    __phase: Phase = Phase.STAGE_ONE
    __random_client_number = None

    def _select_workers(self) -> set:
        self._endpoint.quant_broadcast = True
        if self.__phase != Phase.STAGE_ONE:
            return super()._select_workers()
        self.__random_client_number = self.config.algorithm_kwargs.get(
            "random_client_number", None
        )
        if self.__random_client_number is not None:
            return set(
                random.sample(
                    list(range(self.worker_number)), k=self.__random_client_number
                )
            )
        return super()._select_workers()

    def _aggregate_worker_data(self, worker_data):
        self._send_parameter_diff = False
        self._compute_stat = False
        if self.__phase == Phase.STAGE_ONE:
            self._compute_stat = True
        assert worker_data
        for v in worker_data.values():
            v = v.data
            if "round" in v:
                self._round_number = v["round"]
            if "check_acc" in v:
                self._compute_stat = True
            if "final_aggregation" in v:
                self.__phase = Phase.END
            break
        result = super()._aggregate_worker_data(worker_data=worker_data)
        match self.__phase:
            case Phase.STAGE_ONE:
                if self.round_number >= self.config.round or (
                    self._early_stop and not self._has_improvement()
                ):
                    get_logger().warning("switch to phase 2")
                    self.__phase = Phase.STAGE_TWO
                    # self._endpoint.client_quantized_keys = {"quantized_parameter_diff"}
                    result["phase_two"] = True
                    return result
            case Phase.STAGE_TWO:
                if self._early_stop and not self._has_improvement():
                    get_logger().warning("stop aggregation")
                    self._record_compute_stat(result["parameter"])
                    result["end_training"] = True
                    return result
            case Phase.END:
                return {"end_training": True}
            case _:
                get_logger().warning("unknown phase %s", self.__phase)
                raise NotImplementedError()
        return result

    def _stopped(self) -> bool:
        return self.__phase == Phase.END

    def _has_improvement(self) -> bool:
        if self.__phase == Phase.STAGE_TWO:
            return True
        return not self._convergent()
