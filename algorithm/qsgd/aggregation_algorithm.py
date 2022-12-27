"""QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding  https://arxiv.org/abs/1610.02132"""

from ..aggregation_algorithm import AggregationAlgorithm


class QSGDAlgorithm(AggregationAlgorithm):
    def aggregate_worker_data(self, worker_data: dict, **kwargs: dict) -> dict:
        gradients = {}
        dataset_sizes = {}
        for worker_id, data in worker_data.items():
            data = data.data
            dataset_sizes[worker_id] = data["dataset_size"]
            gradients[worker_id] = data["gradient"]

        return {
            "gradient": AggregationAlgorithm.weighted_avg(
                gradients, AggregationAlgorithm.get_ratios(dataset_sizes)
            )
        }
