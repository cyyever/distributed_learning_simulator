from config import DistributedTrainingConfig
from cyy_torch_toolbox.ml_type import MachineLearningPhase


class DataSplitterBase:
    def __init__(self) -> None:
        self._dataset_indices: dict[MachineLearningPhase, list] = {}

    def get_dataset_indices(self, worker_id: int) -> dict:
        return {
            phase: self._dataset_indices[phase][worker_id]
            for phase in MachineLearningPhase
        }


class IIDSplitter(DataSplitterBase):
    def __init__(self, config: DistributedTrainingConfig):
        super().__init__()
        dc = config.create_dataset_collection()
        parts = [1] * config.worker_number
        for phase in MachineLearningPhase:
            self._dataset_indices[phase] = dc.get_dataset_util(phase).iid_split_indices(
                parts
            )


class RandomSplitter(DataSplitterBase):
    def __init__(self, config: DistributedTrainingConfig):
        super().__init__()
        dc = config.create_dataset_collection()
        parts = [1] * config.worker_number
        for phase in MachineLearningPhase:
            self._dataset_indices[phase] = dc.get_dataset_util(
                phase
            ).random_split_indices(parts)
