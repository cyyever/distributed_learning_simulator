import random

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox import (ClassificationDatasetCollection,
                               DatasetCollection, DatasetCollectionSampler,
                               MachineLearningPhase)


class RandomLabelIIDSplit(DatasetCollectionSampler):
    def __init__(
        self,
        dataset_collection: DatasetCollection,
        part_number: int,
        sampled_class_number: int,
    ) -> None:
        super().__init__(dataset_collection=dataset_collection, part_number=part_number)
        assert isinstance(dataset_collection, ClassificationDatasetCollection)
        assert not dataset_collection.is_mutilabel()
        labels = dataset_collection.get_labels()
        assert sampled_class_number < len(labels)
        assigned_labels = [
            random.sample(list(labels), sampled_class_number)
            for _ in range(part_number)
        ]

        # Assure that all labels are allocated
        assert len(labels) == len(set(sum(assigned_labels, start=[])))

        for phase in MachineLearningPhase:
            self._dataset_indices[phase] = dict(
                enumerate(
                    self._samplers[phase].split_indices(
                        parts=[
                            {label: 1 for label in labels} for labels in assigned_labels
                        ]
                    )
                )
            )
        for worker_id, labels in enumerate(assigned_labels):
            get_logger().info("worker %s has assigned labels %s", worker_id, labels)
            training_set_size = len(
                self._dataset_indices[MachineLearningPhase.Training][worker_id]
            )
            get_logger().info(
                "worker %s has training set size %s", worker_id, training_set_size
            )
