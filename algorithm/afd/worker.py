from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.tensor import cat_tensors_to_vector

from worker.fed_avg_worker import FedAVGWorker


class SingleModelAdaptiveFedDropoutWorker(FedAVGWorker):
    __parameter_keys: set = set()
    __full_parameter_keys = None

    def _get_sent_data(self) -> dict:
        self._send_parameter_diff = False
        self._choose_model_by_validation = False
        sent_data = super()._get_sent_data()
        sent_data |= {
            "training_loss": self.trainer.performance_metric.get_loss(
                self.trainer.hyper_parameter.epoch
            ).item()
        }
        assert self.__parameter_keys
        sent_data["parameter"] = {
            k: v
            for k, v in sent_data["parameter"].items()
            if k in self.__parameter_keys
        }
        get_logger().info(
            "send_num %s",
            cat_tensors_to_vector(sent_data["parameter"].values()).numel(),
        )
        return sent_data

    def _load_result_from_server(self, result):
        assert result["parameter"]
        self.__parameter_keys = set(result["parameter"].keys())
        super()._load_result_from_server(result)
