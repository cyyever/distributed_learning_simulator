from config import DistributedTrainingConfig

from .data_spliting import DataSplitterBase, IIDSplitter, RandomSplitter


def get_data_splitter(config: DistributedTrainingConfig) -> DataSplitterBase:
    match config.dataset_split_method.lower():
        case "iid":
            return IIDSplitter(config)
        case "random":
            return RandomSplitter(config)
    raise NotImplementedError(config.dataset_split_method)
