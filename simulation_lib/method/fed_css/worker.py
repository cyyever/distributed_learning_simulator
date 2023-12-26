from typing import Any

import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.tensor import cat_tensors_to_vector

from ..common_import import Message, ParameterMessageBase
from ..fed_aas.worker import FedAASWorker


class FedCSSWorker(FedAASWorker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__round_cnt = 0

    def _load_result_from_server(self, result: Message) -> None:
        after_aggregation = True
        if not isinstance(result, ParameterMessageBase):
            after_aggregation = False
        super()._load_result_from_server(result=result)
        if not after_aggregation:
            return
        self.__round_cnt += 1
        warmup_rounds = self.config.algorithm_kwargs["warmup_rounds"]
        if self.__round_cnt <= warmup_rounds:
            return
        sample_indices = self._sample_nodes()
        input_nodes = torch.tensor(sample_indices, dtype=torch.long)

        self.trainer.update_dataloader_kwargs(
            pyg_input_nodes={MachineLearningPhase.Training: input_nodes}
        )

    def _sample_nodes(self):
        sample_percent = self.config.algorithm_kwargs["sample_percent"]
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
