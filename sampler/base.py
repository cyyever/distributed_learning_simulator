import random

from config import DistributedTrainingConfig
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.dataset.sampler import DatasetSampler
from cyy_torch_toolbox.ml_type import MachineLearningPhase


class SamplerBase:
    def __init__(self) -> None:
        self._dataset_indices: dict[MachineLearningPhase, list | dict] = {}

    def get_dataset_indices(self, worker_id: int) -> dict:
        return {
            phase: self._dataset_indices[phase][worker_id]
            for phase in MachineLearningPhase
        }


class IIDSampler(SamplerBase):
    def __init__(self, config: DistributedTrainingConfig):
        super().__init__()
        dc = config.create_dataset_collection()
        parts = [1] * config.worker_number
        for phase in MachineLearningPhase:
            self._dataset_indices[phase] = DatasetSampler(
                dc.get_dataset_util(phase)
            ).iid_split_indices(parts)


class RandomSampler(SamplerBase):
    def __init__(self, config: DistributedTrainingConfig):
        super().__init__()
        dc = config.create_dataset_collection()
        parts = [1] * config.worker_number
        for phase in MachineLearningPhase:
            self._dataset_indices[phase] = DatasetSampler(
                dc.get_dataset_util(phase)
            ).random_split_indices(parts)


class RandomClassSampler(SamplerBase):
    def __init__(self, config: DistributedTrainingConfig) -> None:
        super().__init__()
        sampled_class_number = config.algorithm_kwargs["sampled_class_number"]
        dc = config.create_dataset_collection()
        labels = dc.get_dataset_util(MachineLearningPhase.Training).get_labels()
        assert sampled_class_number < len(labels)
        assigned_workers: dict[int, set] = {}
        for worker_id in range(config.worker_number):
            for label in random.sample(list(labels), sampled_class_number):
                if label not in assigned_workers:
                    assigned_workers[label] = set()
                assigned_workers[label].add(worker_id)

        # Assure that all labels are allocated
        assert len(labels) == len(assigned_workers)

        for label, worker_ids in assigned_workers.items():
            get_logger().info("label %s has %s workers", label, len(worker_ids))

        for phase in MachineLearningPhase:
            self._dataset_indices[phase] = {}
        assigned_indices: set[int] = set()
        for phase in MachineLearningPhase:
            sampler = DatasetSampler(dc.get_dataset_util(phase))
            for label, worker_ids in assigned_workers.items():
                worker_index_sets = sampler.iid_split_indices(
                    parts=[1] * len(worker_ids),
                    labels=[label],
                    excluded_indices=assigned_indices,
                )
                for worker_id, worker_index_set in zip(worker_ids, worker_index_sets):
                    if worker_id not in self._dataset_indices[phase]:
                        self._dataset_indices[phase][worker_id] = set()
                    self._dataset_indices[phase][worker_id] |= worker_index_set
                    assigned_indices |= worker_index_set
        total_training_set_size = 0
        for worker_id in range(config.worker_number):
            labels = set()
            for label, worker_ids in assigned_workers.items():
                if worker_id in worker_ids:
                    labels.add(label)
            get_logger().info("worker %s has assigned labels %s", worker_id, labels)
            training_set_size = len(
                self._dataset_indices[MachineLearningPhase.Training][worker_id]
            )
            get_logger().info(
                "worker %s has training set size %s", worker_id, training_set_size
            )
            total_training_set_size += training_set_size
