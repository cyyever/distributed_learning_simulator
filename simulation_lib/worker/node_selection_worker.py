from typing import Any, Iterable, Protocol

import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_torch_toolbox import ExecutorHookPoint, MachineLearningPhase, Trainer
from cyy_torch_toolbox.tensor import cat_tensors_to_vector

from ..config import DistributedTrainingConfig


class GraphWorkerProtocol(Protocol):
    @property
    def trainer(self) -> Trainer:
        ...

    @property
    def round_index(self) -> int:
        ...

    @property
    def config(self) -> DistributedTrainingConfig:
        ...

    @property
    def training_node_indices(self) -> Iterable[int]:
        ...


class NodeSelectionMixin(GraphWorkerProtocol):
    hook_name = "choose_graph_nodes"

    def remove_node_selection_hook(self) -> None:
        self.trainer.remove_named_hook(name=self.hook_name)

    def append_node_selection_hook(self) -> None:
        self.remove_node_selection_hook()
        self.trainer.append_named_hook(
            hook_point=ExecutorHookPoint.BEFORE_EPOCH,
            name=self.hook_name,
            fun=self.__hook_impl,
        )

    def __hook_impl(self, **kwargs: Any) -> None:
        self.update_nodes()

    def update_nodes(self) -> None:
        warmup_rounds: int = self.config.algorithm_kwargs.get("warmup_rounds", 0)
        if self.round_index + 1 <= warmup_rounds:
            return
        sample_indices = self._sample_nodes()
        input_nodes = torch.tensor(sample_indices, dtype=torch.long)

        self.trainer.update_dataloader_kwargs(
            pyg_input_nodes={MachineLearningPhase.Training: input_nodes}
        )

    def _sample_nodes(self) -> list[int]:
        self.trainer.update_dataloader_kwargs(pyg_input_nodes={})
        sample_percent: float = self.config.algorithm_kwargs.get("sample_percent", 1.0)
        if sample_percent >= 1.0:
            return list(self.training_node_indices)
        if self.config.algorithm_kwargs.get("random_selection", False):
            sample_prob = torch.ones(size=(self.trainer.dataset_size,))
            sample_res = torch.multinomial(
                sample_prob,
                int(sample_prob.numel() * sample_percent),
                replacement=False,
            )
            assert sample_res.numel() != 0
            sample_indices = sorted(self.training_node_indices)
            return [sample_indices[idx] for idx in sample_res.tolist()]
        inferencer = self.trainer.get_inferencer(
            phase=MachineLearningPhase.Training, deepcopy_model=False
        )
        if "batch_number" in self.trainer.dataloader_kwargs:
            batch_size = (
                self.trainer.dataset_size
                / self.trainer.dataloader_kwargs["batch_number"]
            )
            inferencer.remove_dataloader_kwargs("batch_number")
            inferencer.update_dataloader_kwargs(batch_size=batch_size)
        inferencer.update_dataloader_kwargs(ensure_batch_size_cover=True)
        sample_loss_dict = inferencer.get_sample_loss()
        sample_indices = sorted(sample_loss_dict.keys())
        sample_loss = cat_tensors_to_vector(
            get_mapping_values_by_key_order(sample_loss_dict)
        )

        sample_prob = sample_loss / sample_loss.sum()
        sample_res = torch.multinomial(
            sample_prob, int(sample_prob.numel() * sample_percent), replacement=False
        )
        assert sample_res.numel() != 0
        return [sample_indices[idx] for idx in sample_res.tolist()]
