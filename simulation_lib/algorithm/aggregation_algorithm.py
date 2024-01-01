from typing import Any

from cyy_torch_toolbox.tensor import tensor_to
from cyy_torch_toolbox.typing import TensorDict

from ..message import DeltaParameterMessage, Message, ParameterMessage


class AggregationAlgorithm:
    def __init__(self) -> None:
        self._all_worker_data: dict[int, Message] = {}
        self.__skipped_workers: set[int] = set()

    @classmethod
    def get_ratios(
        cls, data_dict: dict[int, ParameterMessage], key_name: str | None = None
    ) -> dict[int, float]:
        if key_name is None:
            total_scalar = sum(v.dataset_size for v in data_dict.values())
            return {
                k: float(v.dataset_size) / float(total_scalar)
                for k, v in data_dict.items()
            }
        total_scalar = sum(v.other_data[key_name] for v in data_dict.values())
        return {
            k: float(v.other_data[key_name]) / float(total_scalar)
            for k, v in data_dict.items()
        }

    @classmethod
    def weighted_avg(
        cls,
        data_dict: dict[int, ParameterMessage],
        weight_dict: dict[int, float],
    ) -> TensorDict:
        assert data_dict
        avg_data: TensorDict = {}
        for worker_id, v in data_dict.items():
            ratio = weight_dict[worker_id]
            assert 0 <= ratio <= 1

            d = {k2: v2 * ratio for (k2, v2) in v.parameter.items()}
            if not avg_data:
                avg_data = d
            else:
                for k in avg_data:
                    avg_data[k] += d[k]
        for p in avg_data.values():
            assert not p.isnan().any().cpu()
        return avg_data

    def __process_worker_data(
        self,
        worker_data: Message,
        old_parameter_dict: TensorDict | None,
    ) -> Message:
        match worker_data:
            case DeltaParameterMessage():
                assert old_parameter_dict is not None
                worker_data.delta_parameter = tensor_to(
                    worker_data.delta_parameter, device="cpu"
                )
                return worker_data.restore(old_parameter_dict)
            case ParameterMessage():
                if old_parameter_dict is not None:
                    worker_data.complete(old_parameter_dict)
                worker_data.parameter = tensor_to(worker_data.parameter, device="cpu")
                return worker_data
            case Message():
                return worker_data
        raise NotImplementedError(worker_data)

    def process_worker_data(
        self,
        worker_id: int,
        worker_data: Message | None,
        old_parameter_dict: TensorDict | None,
        save_dir: str,
    ) -> None:
        if worker_data is None:
            self.__skipped_workers.add(worker_id)
            return
        self._all_worker_data[worker_id] = self.__process_worker_data(
            worker_data=worker_data,
            old_parameter_dict=old_parameter_dict,
        )

    def aggregate_worker_data(self) -> Any:
        raise NotImplementedError()

    def clear_worker_data(self) -> None:
        self._all_worker_data.clear()
        self.__skipped_workers.clear()

    def exit(self) -> None:
        pass
