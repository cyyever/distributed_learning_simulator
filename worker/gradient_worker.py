import json
import os

import torch
from cyy_torch_toolbox.ml_type import ExecutorHookPoint
from cyy_torch_toolbox.tensor import (cat_tensors_to_vector,
                                      decompose_tensor_to_list)
from torch.optim.sgd import SGD

from .client import Client


def compute_gradient(
    params,
    d_p_list,
    weight_decay: float,
) -> list:
    real_gradient = []
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)
        real_gradient.append(d_p)

    assert len(real_gradient) == len(params)
    return real_gradient


class GradientWorker(Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self.trainer.get_optimizer(), SGD)
        self.__epoch_stat = {}
        self.trainer.append_named_hook(
            ExecutorHookPoint.OPTIMIZER_STEP, "step", self.__step
        )
        self.trainer.append_named_hook(
            ExecutorHookPoint.AFTER_EPOCH, "record", self.__record
        )
        self.trainer.append_named_hook(
            ExecutorHookPoint.AFTER_EXECUTE, "report_end", self.__report_end
        )

    def __report_end(self, **kwargs):
        self.send_data_to_server({"end_training": True})

    def _process_gradient(self, gradient):
        return gradient

    def __step(self, executor):
        trainer = executor
        optimizer = trainer.get_optimizer()
        if hasattr(optimizer, "_step_count"):
            optimizer._step_count += 1

        assert len(optimizer.param_groups) == 1
        for group in optimizer.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            with torch.no_grad():
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]
                lr = group["lr"]
                for p in group["params"]:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        d_p_list.append(p.grad)

                        state = optimizer.state[p]
                        if "momentum_buffer" not in state:
                            momentum_buffer_list.append(None)
                        else:
                            momentum_buffer_list.append(state["momentum_buffer"])

                gradient_list = compute_gradient(
                    params_with_grad,
                    d_p_list,
                    weight_decay,
                )
            gradient_shape = [p.shape for p in gradient_list]

            gradient = self._process_gradient(cat_tensors_to_vector(gradient_list))
            self.send_data_to_server(
                {"dataset_size": self.trainer.dataset_size, "gradient": gradient}
            )
            gradient = self._get_result_from_server()["gradient"]
            gradient_list = decompose_tensor_to_list(
                shapes=gradient_shape, tensor=gradient
            )

            with torch.no_grad():
                for d_p, param, momentum_buffer in zip(
                    gradient_list, params_with_grad, momentum_buffer_list
                ):
                    d_p = d_p.to(param.device)
                    if momentum != 0:
                        buf = momentum_buffer
                        if buf is None:
                            buf = d_p.detach()
                            momentum_buffer = buf
                        else:
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf
                    param.add_(d_p, alpha=-lr)
                    # update momentum_buffers in state
                    state = optimizer.state[param]
                    state["momentum_buffer"] = momentum_buffer

    def __record(self, **kwargs):
        epoch = kwargs["epoch"]
        trainer = kwargs["executor"]
        self.__epoch_stat[epoch] = {}
        self.__epoch_stat[epoch]["loss"] = trainer.performance_metric.get_loss(
            epoch
        ).data.item()
        self.__epoch_stat[epoch]["accuracy"] = trainer.performance_metric.get_accuracy(
            epoch
        ).data.item()
        with open(
            os.path.join(self.save_dir, "epoch_stat.json"), "wt", encoding="utf8"
        ) as f:
            json.dump(self.__epoch_stat, f)
