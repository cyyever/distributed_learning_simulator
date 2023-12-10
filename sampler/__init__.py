import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(currentdir, ".."))

from config import DistributedTrainingConfig

from .base import IIDSampler, RandomClassSampler, RandomSampler, SamplerBase


def get_dataset_sampler(config: DistributedTrainingConfig) -> SamplerBase:
    match config.dataset_sampling.lower():
        case "iid":
            return IIDSampler(config)
        case "random":
            return RandomSampler(config)
        case "random_class":
            return RandomClassSampler(config)
    raise NotImplementedError(config.dataset_sampling)
