from typing import Any

from cyy_torch_toolbox import ExecutorHookPoint, MachineLearningPhase
from distributed_learning_simulation import (GraphWorker, Message,
                                             NodeSelectionMixin)


class FedGNNWorker(GraphWorker, NodeSelectionMixin):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if (
            len(
                self.trainer.dataset_collection.get_dataset_util(
                    MachineLearningPhase.Validation
                )
            )
            < self.trainer.dataloader_kwargs["batch_number"]
        ):
            self.disable_choosing_model_by_validation()

    def _load_result_from_server(self, result: Message) -> None:
        super()._load_result_from_server(result=result)
        hook_name = "choose_nodes"
        self.trainer.remove_named_hook(name=hook_name)

        def __hook_impl(**kwargs: Any) -> None:
            self.update_nodes()

        self.trainer.append_named_hook(
            hook_point=ExecutorHookPoint.BEFORE_EPOCH,
            name=hook_name,
            fun=__hook_impl,
        )
