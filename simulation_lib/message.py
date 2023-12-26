import copy
from dataclasses import dataclass, field, fields
from typing import Any

import torch
from cyy_torch_toolbox.tensor import recursive_tensor_op
from cyy_torch_toolbox.typing import TensorDict


@dataclass(kw_only=True)
class Message:
    other_data: dict = field(default_factory=lambda: {})
    in_round: bool = False
    end_training: bool = False


@dataclass(kw_only=True)
class ParameterMessageBase(Message):
    dataset_size: int = 0


@dataclass(kw_only=True)
class ParameterMessage(ParameterMessageBase):
    parameter: TensorDict

    def complete(self, other_parameter: TensorDict) -> None:
        for k, v in other_parameter.items():
            if k not in self.parameter:
                self.parameter[k] = v


@dataclass(kw_only=True)
class ParameterFileMessage(ParameterMessageBase):
    path: str


@dataclass(kw_only=True)
class DeltaParameterMessage(ParameterMessageBase):
    delta_parameter: TensorDict

    def restore(self, parameter: TensorDict) -> ParameterMessage:
        new_parameter = copy.deepcopy(parameter)
        for k, v in self.delta_parameter.items():
            new_parameter[k] += v
        msg = ParameterMessage(parameter=new_parameter)
        for f in fields(self):
            setattr(msg, f.name, getattr(self, f.name))
        msg.parameter = new_parameter
        return msg


def get_message_size(msg: Message) -> int:
    cnt: int = 0

    def count(data: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        nonlocal cnt
        cnt += data.element_size() * data.numel()
        return data

    recursive_tensor_op(msg, fun=count)
    assert cnt > 0
    return cnt
