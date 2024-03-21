from ..common_import import (AggregationServer, Message, MultipleWorkerMessage,
                             ParameterMessage)


class GraphFedServer(AggregationServer):
    def _before_send_result(self, result: Message) -> None:
        if isinstance(result, MultipleWorkerMessage):
            parameter = result.other_data.pop("centralized_parameter", None)
            if parameter is not None:
                super()._before_send_result(
                    result=ParameterMessage(parameter=parameter)
                )
                return
        super()._before_send_result(result=result)
