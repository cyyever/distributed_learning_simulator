""" signSGD: Compressed Optimisation for Non-Convex Problems https://arxiv.org/abs/1802.04434 """
from cyy_torch_toolbox.typing import TensorDict

from ..common_import import GradientWorker, ParameterMessage


class SignSGDWorker(GradientWorker):
    def _process_gradient(self, gradient_dict: TensorDict) -> TensorDict:
        self.send_data_to_server(
            ParameterMessage(
                parameter={k: v.sign() for k, v in gradient_dict.items()},
                in_round=True,
                aggregation_weight=self.trainer.dataset_size,
            )
        )
        result = self._get_data_from_server()
        assert isinstance(result, ParameterMessage)
        return result.parameter
