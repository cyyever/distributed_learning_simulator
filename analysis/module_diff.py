import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.model_util import ModelUtil
from cyy_torch_toolbox.tensor import cat_tensors_to_vector


class ModuleDiff(Hook):
    def __init__(self, delta=0.1):
        super().__init__()
        self.__prev_modules = None
        self.__delta = delta

    def __get_sub_modules(self, model_executor):
        model_util = ModelUtil(model_executor.model)
        sub_modules = []
        for module_name, module in model_util.get_sub_modules():
            if len(list(module.parameters())) == 0:
                # get_logger().warning("ignore module %s", module_name)
                continue
            sub_modules.append(
                (module_name, cat_tensors_to_vector(module.parameters()).cpu().detach())
            )
        return sub_modules

    def _before_execute(self, **kwargs):
        assert self.__prev_modules is None
        model_executor = kwargs["model_executor"]
        self.__prev_modules = self.__get_sub_modules(model_executor)

    def _after_load_model(self, **kwargs):
        model_executor = kwargs["model_executor"]
        cur_modules = self.__get_sub_modules(model_executor)
        assert len(self.__prev_modules) == len(cur_modules)
        for (prev_sub_module_name, prev_parameter), (
            sub_module_name,
            parameter,
        ) in zip(self.__prev_modules, cur_modules):
            assert prev_sub_module_name == sub_module_name
            diff = torch.linalg.norm(prev_parameter - parameter)
            if self.__delta is not None and diff <= self.__delta:
                continue
            get_logger().info("module %s has diff %s", sub_module_name, diff)
        self.__prev_modules = cur_modules
