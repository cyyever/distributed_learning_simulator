from typing import Any, Callable

import torch
from cyy_torch_toolbox.ml_type import ExecutorHookPoint
from cyy_torch_toolbox.model.evaluator import ModelEvaluator
from cyy_torch_toolbox.tensor import tensor_to
from cyy_torch_toolbox.typing import TensorDict

from ..message import Message, ParameterMessage
from .client import Client


class GradientModelEvaluator:
    def __init__(
        self,
        evaluator: ModelEvaluator,
        gradient_fun: Callable,
        aggregation_indicator_fun: Callable,
    ) -> None:
        assert torch.cuda.is_available()
        self.evaluator: ModelEvaluator = evaluator
        self.__gradient_fun: Callable = gradient_fun
        self.__aggregation_indicator_fun: Callable = aggregation_indicator_fun

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.evaluator.__call__(*args, **kwargs)

    def __getattr__(self, name):
        if name in ("evaluator", "gradient_fun"):
            raise AttributeError()
        return getattr(self.evaluator, name)

    def backward_and_step(
        self,
        loss,
        optimizer: torch.optim.Optimizer,
        **backward_kwargs,
    ) -> None:
        self.backward(loss=loss, optimizer=optimizer, **backward_kwargs)
        optimizer.step()

    def backward(
        self,
        loss,
        optimizer: torch.optim.Optimizer,
        **backward_kwargs,
    ) -> Any:
        self.evaluator.backward(loss=loss, optimizer=optimizer, **backward_kwargs)
        if not self.__aggregation_indicator_fun():
            return
        gradient_dict: TensorDict = self.__gradient_fun(
            self.evaluator.model_util.get_gradient_dict()
        )
        self.evaluator.model_util.load_gradient_dict(
            tensor_to(gradient_dict, device=loss.device)
        )


class GradientWorker(Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__cnt = 0
        self.__aggregation_interval = self.config.algorithm_kwargs.get("interval", 1)
        self.trainer.replace_model_evaluator(
            lambda evaluator: GradientModelEvaluator(
                evaluator=evaluator,
                gradient_fun=self._process_gradient,
                aggregation_indicator_fun=self._should_aggregate,
            )
        )
        self.trainer.append_named_hook(
            ExecutorHookPoint.AFTER_EXECUTE, "end_training", self.__report_end
        )

    def __report_end(self, **kwargs: Any) -> None:
        self.send_data_to_server(Message(end_training=True))

    def _should_aggregate(self) -> bool:
        res = self.__cnt % self.__aggregation_interval
        self.__cnt += 1
        return res

    def _process_gradient(self, gradient_dict: TensorDict) -> TensorDict:
        self.send_data_to_server(
            ParameterMessage(
                parameter=gradient_dict,
                in_round=True,
                dataset_size=self.trainer.dataset_size,
            )
        )
        result = self._get_data_from_server()
        assert isinstance(result, ParameterMessage)
        return result.parameter
