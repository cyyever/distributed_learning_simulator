from cyy_naive_lib.storage import DataStorage

from .aggregation_algorithm import AggregationAlgorithm


class FedAVGAlgorithm(AggregationAlgorithm):
    def aggregate_worker_data(
        self, worker_data: dict[str, DataStorage], old_parameter_dict: dict | None
    ) -> dict:
        result = self.extract_data(
            worker_data=worker_data, old_parameter_dict=old_parameter_dict
        )
        avg_parameter = AggregationAlgorithm.weighted_avg(
            result["parameter"], AggregationAlgorithm.get_ratios(result["dataset_size"])
        )
        return avg_parameter
