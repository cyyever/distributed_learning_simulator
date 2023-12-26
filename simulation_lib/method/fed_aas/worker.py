from typing import Any

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.ml_type import ExecutorHookPoint

from ..common_import import GraphWorker, Message


class FedAASWorker(GraphWorker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert self._share_feature
        self.__sharing_interval = None

    def _before_training(self) -> None:
        super()._before_training()
        self.trainer.append_named_hook(
            ExecutorHookPoint.BEFORE_BATCH,
            "embedding_sharing",
            self.__embedding_sharing,
        )

    def _load_result_from_server(self, result: Message) -> None:
        if "sharing_interval" in result.other_data:
            self.__sharing_interval = result.other_data.pop("sharing_interval")
        super()._load_result_from_server(result=result)

    def __embedding_sharing(self, batch_index: int, **kwargs: Any) -> None:
        assert self.__sharing_interval is not None
        if batch_index % self.__sharing_interval == 0:
            if self.worker_id == 0:
                get_logger().error(
                    "share embedding in batch %s %s",
                    batch_index,
                    self.__sharing_interval,
                )
            hook = self._pass_node_feature
        else:
            hook = self._clear_cross_client_edge_on_the_fly
        assert len(self._hook_handles) > 1
        for idx in range(1, len(self._hook_handles)):
            self._register_embedding_hook(module_index=idx, hook=hook)
