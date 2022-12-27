import os
from typing import TypeAlias

import torch
from cyy_naive_lib.storage import DataStorage
from cyy_torch_toolbox.tensor import tensor_to

ParameterDictType: TypeAlias = dict[str, torch.Tensor]


class ModelCache:
    def __init__(self):
        self.__parameter_dict: DataStorage = DataStorage()

    @property
    def cached_parameter_dict(self) -> ParameterDictType:
        return self.__parameter_dict.data

    def cache_parameter_dict(self, parameter_dict: ParameterDictType) -> None:
        self.__parameter_dict.set_data(tensor_to(parameter_dict, device="cpu"))

    def discard(self):
        self.__parameter_dict.set_data(None)

    def get_parameter_diff(
        self, new_parameter_dict: ParameterDictType
    ) -> ParameterDictType:
        return {
            k: tensor_to(v, device="cpu") - self.cached_parameter_dict[k]
            for k, v in new_parameter_dict.items()
        }

    def add_parameter_diff(self, parameter_diff: ParameterDictType) -> None:
        for k, v in self.cached_parameter_dict.items():
            self.cached_parameter_dict[k] = v + tensor_to(
                parameter_diff[k], device="cpu"
            )
        self.__parameter_dict.mark_new_data()

    def save(self, save_dir: str) -> None:
        self.__parameter_dict.set_data_path(os.path.join(save_dir, "cached_model.pk"))
        self.__parameter_dict.save()
