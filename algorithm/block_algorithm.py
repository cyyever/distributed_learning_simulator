import torch
import torch.nn
from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import BlockType, ModelUtil
from cyy_torch_toolbox.tensor import cat_tensors_to_vector
from distributed_learning_simulation.worker.protocol import AggregationWorkerProtocol


class BlockAlgorithmMixin(AggregationWorkerProtocol):
    def __init__(self) -> None:
        super().__init__()
        self.__blocks: list[BlockType] | None = None
        self._block_types = {
            ("AlbertTransformer",),
            ("AlbertEmbeddings",),
            ("Bottleneck",),
            ("TransformerEncoderLayer",),
            (torch.nn.BatchNorm2d, torch.nn.ReLU, torch.nn.Conv2d),
            (torch.nn.BatchNorm2d, torch.nn.Conv2d),
            (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        }

    @property
    def blocks(self) -> list[BlockType]:
        if self.__blocks is None:
            self._find_blocks()
        assert self.__blocks is not None
        return self.__blocks

    def _get_model_util(self) -> ModelUtil:
        return self.trainer.model_util

    def _find_blocks(self) -> None:
        model_util = self._get_model_util()
        blocks = model_util.get_module_blocks(block_types=self._block_types)
        self.__blocks = []
        modules = list(model_util.get_modules())
        while modules:
            submodule_name, submodule = modules[0]
            del modules[0]
            if not submodule_name:
                continue
            if len(list(submodule.parameters())) == 0:
                continue
            part_of_block = False
            in_block = False
            tmp_blocks = []

            if blocks:
                tmp_blocks.append(blocks[0])
            if self.__blocks:
                tmp_blocks.append(self.__blocks[-1])
            for block in tmp_blocks:
                for block_submodule_name, _ in block:
                    if block_submodule_name == submodule_name:
                        part_of_block = True
                        self.__blocks.append(block)
                        for _ in range(len(block) - 1):
                            del modules[0]
                        del blocks[0]
                        break
                    if submodule_name.startswith(
                        f"{block_submodule_name}."
                    ) or block_submodule_name.startswith(f"{submodule_name}."):
                        in_block = True
                        break
                if part_of_block or in_block:
                    break
            if part_of_block or in_block:
                continue
            self.__blocks.append([(submodule_name, submodule)])
            if self.hold_log_lock:
                log_info("identify a submodule:%s", submodule_name)

        if self.hold_log_lock:
            log_info("identify these blocks in model:")
            for block in self.__blocks:
                log_info(
                    "%s",
                    [f"{name}" for name, _ in block],
                )

        # check the parameter numbers are the same
        tmp_parameter_list = []
        tmp_parameter_name = set()
        for block in self.__blocks:
            for submodule_name, submodule in block:
                for p_name, p in submodule.named_parameters():
                    tmp_parameter_list.append(p)
                    if submodule_name:
                        tmp_parameter_name.add(submodule_name + "." + p_name)
                    else:
                        tmp_parameter_name.add(p_name)
        parameter_dict = model_util.get_parameters()
        if tmp_parameter_name != set(parameter_dict.keys()):
            for a in tmp_parameter_name:
                if a not in parameter_dict:
                    raise RuntimeError(a + " not in model")
            for a in parameter_dict:
                if a not in tmp_parameter_name:
                    raise RuntimeError(a + " not in block")
        parameter_list = model_util.get_parameter_list()
        assert cat_tensors_to_vector(tmp_parameter_list).shape == parameter_list.shape
