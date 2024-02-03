import random
from typing import Any

import torch
from cyy_naive_lib.algorithm.mapping_op import (
    get_mapping_items_by_key_order, get_mapping_values_by_key_order)
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox import (ClassificationDatasetCollection,
                               DatasetCollection, DatasetCollectionSplit,
                               MachineLearningPhase, SplitBase)
from cyy_torch_toolbox.dataset import (  # noqa: F401
    get_dataset_collection_sampler, get_dataset_collection_split,
    global_sampler_factory)


class RandomLabelIIDSplit(SplitBase):
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
                        part_proportions=[
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


class DirichletSplit(DatasetCollectionSplit):
    def __init__(
        self,
        dataset_collection: ClassificationDatasetCollection,
        concentration: float | list[dict[Any, float]],
        part_number: int,
    ) -> None:
        if not isinstance(concentration, list):
            all_labels = dataset_collection.get_labels()
            concentration = [
                {label: float(concentration) for label in all_labels}
            ] * part_number
        assert isinstance(concentration, list)
        assert len(concentration) == part_number
        part_proportions: list[dict] = []
        for worker_concentration in concentration:
            concentration_tensor = torch.tensor(
                list(get_mapping_values_by_key_order(worker_concentration))
            )
            prob = torch.distributions.dirichlet.Dirichlet(
                concentration_tensor
            ).sample()
            part_proportions.append({})
            for (k, _), label_prob in zip(
                get_mapping_items_by_key_order(worker_concentration), prob
            ):
                part_proportions[-1][k] = label_prob

        super().__init__(
            dataset_collection=dataset_collection, part_proportions=part_proportions
        )


global_sampler_factory.register("random_label_iid", RandomLabelIIDSplit)
global_sampler_factory.register("dirichlet_split", DirichletSplit)
