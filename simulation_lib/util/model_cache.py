from cyy_naive_lib.storage import DataStorage
from cyy_torch_toolbox.tensor import tensor_to
from cyy_torch_toolbox.typing import TensorDict


class ModelCache:
    def __init__(self) -> None:
        self.__parameter_dict: DataStorage = DataStorage()

    @property
    def has_data(self) -> bool:
        return self.__parameter_dict.has_data()

    @property
    def parameter_dict(self) -> TensorDict:
        return self.__parameter_dict.data

    def load_file(self, path: str) -> None:
        self.__parameter_dict = DataStorage(data_path=path)

    def cache_parameter_dict(self, parameter_dict: TensorDict, path: str) -> None:
        self.__parameter_dict.set_data(tensor_to(parameter_dict, device="cpu"))
        self.__parameter_dict.set_data_path(path)

    def get_parameter_diff(self, new_parameter_dict: TensorDict) -> TensorDict:
        return {
            k: tensor_to(v, device="cpu") - self.parameter_dict[k]
            for k, v in new_parameter_dict.items()
        }

    def add_parameter_diff(self, parameter_diff: TensorDict, path: str) -> None:
        self.__parameter_dict.set_data_path(path)
        for k, v in self.parameter_dict.items():
            self.parameter_dict[k] = v + tensor_to(parameter_diff[k], device="cpu")
        self.__parameter_dict.mark_new_data()

    def discard(self) -> None:
        self.__parameter_dict.clear()

    def save(self) -> None:
        self.__parameter_dict.save()

    def get_parameter_path(self) -> str:
        self.__parameter_dict.save()
        assert self.__parameter_dict.data_path
        return self.__parameter_dict.data_path
