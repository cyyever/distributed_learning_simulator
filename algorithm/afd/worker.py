from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.tensor import cat_tensors_to_vector

from worker.fed_avg_worker import FedAVGWorker


class SingleModelAdaptiveFedDropoutWorker(FedAVGWorker):
    __parameter_keys: set = set()

    def _get_sent_data(self) -> dict:
        assert self._choose_model_by_validation
        self._send_parameter_diff = False
        sent_data = super()._get_sent_data()
        sent_data["training_loss"] = self.trainer.best_model["performance_metric"][
            MachineLearningPhase.Training
        ]["loss"]

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
        self.__parameter_keys = set(result["partial_parameter"].keys())
        super()._load_result_from_server(result)
