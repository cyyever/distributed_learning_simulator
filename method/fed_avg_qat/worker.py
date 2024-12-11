import torch.ao.quantization
from cyy_naive_lib.log import log_error
from cyy_torch_algorithm.quantization.qat import QuantizationAwareTraining
from cyy_torch_toolbox import TensorDict
from distributed_learning_simulation import (
    AggregationWorker,
)
from torch.ao.nn.quantized.modules import Linear


class QATWorker(AggregationWorker):
    parameter_name_set: set = set()

    def _get_parameters(self) -> TensorDict:
        assert isinstance(self.trainer.model, torch.ao.quantization.QuantWrapper)
        self.trainer.model.eval()
        old_model = self.trainer.model
        model_int8 = torch.ao.quantization.convert(old_model.module.cpu())
        self.trainer.replace_model(lambda *args: model_int8)
        new_state_dict = {}
        for name, p in self.trainer.model_util.get_modules():
            if isinstance(p, Linear):
                weight, bias = p._packed_params._weight_bias()
                weight_name = name + ".weight"
                assert weight_name in self.parameter_name_set
                new_state_dict[weight_name] = weight.detach().dequantize()
                bias_name = name + ".bias"
                assert bias_name in self.parameter_name_set
                new_state_dict[bias_name] = bias.detach().dequantize()

        for name, p in self.trainer.model.state_dict().items():
            log_error("%s %s", name, p)
            if name.endswith(".zero_point") or name.endswith(".scale"):
                continue
            if name.startswith("module."):
                name = name[len("module.") :]
            if isinstance(p, torch.Tensor | torch.nn.Parameter):
                new_state_dict[name] = p.detach().dequantize()
        log_error("%s %s", new_state_dict.keys(), self.parameter_name_set)
        assert sorted(new_state_dict.keys()) == sorted(self.parameter_name_set)
        if self._model_loading_fun is None:
            self._model_loading_fun = self.load_model
        return new_state_dict

    def load_model(self, state_dict) -> None:
        self.trainer.remove_model()
        self.trainer.model.load_state_dict(state_dict)

    def _before_training(self) -> None:
        super()._before_training()
        self.parameter_name_set = set(self.trainer.model_util.get_parameters().keys())
        self.trainer.append_hook(QuantizationAwareTraining(), "QAT")
