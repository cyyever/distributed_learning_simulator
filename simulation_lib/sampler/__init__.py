
from cyy_torch_toolbox.dataset import (  # noqa: F401
    get_dataset_collection_sampler, global_sampler_factory)

from .base import RandomLabelIIDSplit

global_sampler_factory.register("random_label_iid", RandomLabelIIDSplit)
