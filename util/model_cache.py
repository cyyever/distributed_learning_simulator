import os
import pickle
from typing import TypeAlias

import torch

ParameterDictType: TypeAlias = dict[str, torch.Tensor]


class ModelCache:
    def __init__(self):
        self.__parameter_dict: ParameterDictType = None
        self.__in_disk = False

    @property
    def cached_parameter_dict(self) -> ParameterDictType:
        return self.__parameter_dict

    def cache_parameter_dict(self, parameter_dict: ParameterDictType) -> None:
        self.__parameter_dict = parameter_dict

    def get_parameter_diff(
        self, new_parameter_dict: ParameterDictType
    ) -> ParameterDictType:
        assert self.__parameter_dict
        return {
            k: v - self.cached_parameter_dict[k] for k, v in new_parameter_dict.items()
        }

    def add_parameter_diff(self, parameter_diff: ParameterDictType) -> None:
        assert self.__parameter_dict
        for k, v in self.__parameter_dict.items():
            self.__parameter_dict[k] = v + parameter_diff[k]

    def _load_cached_model_to_memory(self, save_dir: str) -> None:
        if self.__parameter_dict is None and self.__in_disk:
            with open(os.path.join(save_dir, "cached_model.pk"), "rb") as f:
                self.cache_parameter_dict(pickle.load(f))
                self.__in_disk = False

    def _offload_cached_model_from_memory(self, save_dir: str) -> None:
        if self.__parameter_dict is not None:
            with open(os.path.join(save_dir, "cached_model.pk"), "wb") as f:
                pickle.dump(self.cached_parameter_dict, f)
                self.__parameter_dict = None
                self.__in_disk = True
