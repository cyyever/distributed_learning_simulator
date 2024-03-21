import copy
import os

from cyy_torch_toolbox.dataset import get_dataset_collection_sampler

from ..config import load_config_from_file
from ..practitioner import Practitioner
from ..training import get_training_result, train


def test_concurrent_training() -> None:
    config = load_config_from_file(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "conf",
            "fed_avg",
            "mnist.yaml",
        )
    )
    config.hyper_parameter_config.epoch = 1
    config.round = 1
    config.worker_number = 3
    practitioners = set()
    sampler = get_dataset_collection_sampler(
        name=config.dataset_sampling,
        dataset_collection=config.create_dataset_collection(),
        part_number=config.worker_number,
        **config.dataset_sampling_kwargs
    )
    for practitioner_id in range(config.worker_number):
        practitioner = Practitioner(
            practitioner_id=practitioner_id,
        )
        practitioner.set_sampler(sampler=sampler)
        practitioners.add(practitioner)
    task_ids = set()
    for _ in range(5):
        task_id = train(config=config, practitioners=practitioners)
        assert task_id is not None
        task_ids.add(task_id)
    while task_ids:
        for task_id in copy.copy(task_ids):
            get_training_result(task_id=task_id)
            task_ids.remove(task_id)
