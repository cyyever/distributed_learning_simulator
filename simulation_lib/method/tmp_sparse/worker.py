from ..common_import import DeltaParameterMessage, ErrorFeedbackWorker


class FedSparseWorker(ErrorFeedbackWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_percent = self.config.algorithm_kwargs["top_percent"]
        self.error_mom_efficient = self.config.algorithm_kwargs["error_mom_efficient"]

    def sparsify(self, sent_data: DeltaParameterMessage) -> DeltaParameterMessage:
        for name, param in sent_data.delta_parameter.items():
            old_shape = param.shape
            param = param.view(-1)
            if name not in self._error:
                self._error[name] = param.clone()
            else:
                new_param = param + self._error[name]
                self._error[name] = (
                    self.error_mom_efficient * self._error[name] + param.clone()
                )
                param = new_param
            # mask out smallest
            k = int(param.numel() * (1 - self.top_percent))
            _, indices = param.abs().topk(k=k, largest=False, sorted=False)
            sent_data.delta_parameter[name] = param.index_fill(0, indices, 0).reshape(
                old_shape
            )
        return sent_data
