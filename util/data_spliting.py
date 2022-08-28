from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.dataset import sub_dataset
from cyy_torch_toolbox.dataset_transform.transforms import replace_target
from cyy_torch_toolbox.ml_type import MachineLearningPhase, TransformType


class DataSplitter:
    def __init__(self, config):
        self.__config = config
        dc = config.create_dataset_collection()
        parts = [1] * config.worker_number
        self.__dataset_indices = {}
        for phase in MachineLearningPhase:
            if config.iid:
                self.__dataset_indices[phase] = dc.get_dataset_util(
                    phase
                ).iid_split_indices(parts)
            else:
                get_logger().debug("use non IID training dataset")
                self.__dataset_indices[phase] = dc.get_dataset_util(
                    phase
                ).random_split_indices(parts)

    def split(self, trainer, worker_id):
        for phase in MachineLearningPhase:
            trainer.dataset_collection.transform_dataset(
                phase,
                lambda dataset, _: sub_dataset(
                    dataset, self.__dataset_indices[phase][worker_id]
                ),
            )
        if self.__config.noise_percents is not None:
            noise_percent = self.__config.noise_percents[worker_id]
            assert noise_percent != 0
            get_logger().warning("use noise_percent %s", self.__config.noise_percent)
            label_map = trainer.dataset_collection.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).randomize_subset_label(noise_percent)
            trainer.dataset_collection.append_transform(
                transform=replace_target(label_map),
                key=TransformType.Target,
                phases=[MachineLearningPhase.Training],
            )
