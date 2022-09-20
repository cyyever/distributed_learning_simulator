import os
from typing import TypeAlias

import torch
from cyy_naive_lib.storage import DataStorage

ParameterDictType: TypeAlias = dict[str, torch.Tensor]


class ModelCache:
    def __init__(self):
        self.__parameter_dict: DataStorage = DataStorage()

    @property
    def cached_parameter_dict(self) -> ParameterDictType:
        return self.__parameter_dict.data

    def cache_parameter_dict(self, parameter_dict: ParameterDictType) -> None:
        self.__parameter_dict.set_data(parameter_dict)

    def get_parameter_diff(
        self, new_parameter_dict: ParameterDictType
    ) -> ParameterDictType:
        return {
            k: v - self.cached_parameter_dict[k] for k, v in new_parameter_dict.items()
        }

    def add_parameter_diff(self, parameter_diff: ParameterDictType) -> None:
        for k, v in self.cached_parameter_dict.items():
            self.cached_parameter_dict[k] = v + parameter_diff[k]
        self.__parameter_dict.mark_new_data()

    def _save(self, save_dir: str) -> None:
        self.__parameter_dict.set_data_path(os.path.join(save_dir, "cached_model.pk"))
        self.__parameter_dict.save()
