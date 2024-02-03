import math

import torch
from cyy_naive_lib.topology.cs_endpoint import ClientEndpoint

from ..message import FeatureMessage


class DifferentialPrivacyEmbeddingEndpoint(ClientEndpoint):
    def __init__(self, **kwargs) -> None:
        C: float = kwargs.pop("C", 1)
        delta: float = kwargs.pop("delta")
        epsilon: float = kwargs.pop("epsilon")
        super().__init__(**kwargs)
        self.C = C
        self.sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    def send(self, data) -> None:
        if isinstance(data, FeatureMessage) and data.feature is not None:
            for i in range(data.feature.shape[0]):
                f = data.feature[i] / max(1, data.feature[i].norm() / self.C)
                std = torch.zeros_like(f)
                std.fill_((self.sigma * self.C) ** 2)
                data.feature[i] = f + torch.normal(mean=0, std=std)
        super().send(data=data)
