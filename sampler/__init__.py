import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(currentdir, ".."))

from cyy_torch_toolbox.dataset import global_sampler_factory

from .base import RandomClassSampler

global_sampler_factory.register("random_class", RandomClassSampler)
