import random

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox import (DatasetCollection, DatasetCollectionSampler,
                               MachineLearningPhase)


class RandomClassSampler(DatasetCollectionSampler):
    def __init__(
        self,
        dataset_collection: DatasetCollection,
        part_number: int,
        sampled_class_number: int,
    ) -> None:
        super().__init__(dataset_collection=dataset_collection)
        labels = set(
            self._samplers[MachineLearningPhase.Training].label_sample_dict.keys()
        )
        assert sampled_class_number < len(labels)
        assigned_workers: dict[int, set] = {}
        for worker_id in range(part_number):
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
            for label, worker_ids in assigned_workers.items():
                worker_index_sets = self._samplers[phase].iid_split_indices(
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
        for worker_id in range(part_number):
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
