from typing import TypeAlias

import torch
from cyy_naive_lib.storage import DataStorage
from cyy_torch_toolbox.tensor import tensor_to

ParameterDictType: TypeAlias = dict[str, torch.Tensor]


class ModelCache:
    def __init__(self) -> None:
        self.__parameter_dict: DataStorage = DataStorage()

    @property
    def parameter_dict(self) -> ParameterDictType:
        return self.__parameter_dict.data

    def load_file(self, path: str) -> None:
        self.__parameter_dict = DataStorage(data_path=path)

    def cache_parameter_dict(
        self, parameter_dict: ParameterDictType, path: str
    ) -> None:
        self.__parameter_dict.set_data(tensor_to(parameter_dict, device="cpu"))
        self.__parameter_dict.set_data_path(path)

    def discard(self):
        self.__parameter_dict.set_data(None)

    def get_parameter_diff(
        self, new_parameter_dict: ParameterDictType
    ) -> ParameterDictType:
        return {
            k: tensor_to(v, device="cpu") - self.parameter_dict[k]
            for k, v in new_parameter_dict.items()
        }

    def add_parameter_diff(self, parameter_diff: ParameterDictType, path: str) -> None:
        self.__parameter_dict.save()
        self.__parameter_dict.set_data_path(path)
        for k, v in self.parameter_dict.items():
            self.parameter_dict[k] = v + tensor_to(parameter_diff[k], device="cpu")
        self.__parameter_dict.mark_new_data()

    def save(self) -> None:
        self.__parameter_dict.save()

    def get_parameter_path(self) -> str:
        self.__parameter_dict.save()
        assert self.__parameter_dict.data_path
        return self.__parameter_dict.data_path
