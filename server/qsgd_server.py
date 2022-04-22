"""QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding  https://arxiv.org/abs/1610.02132"""

from .aggregation_server import AggregationServer


class QSGDServer(AggregationServer):
    def _aggregate_worker_data(self, round_number, worker_data):
        parameters = {}
        dataset_sizes = {}
        for worker_id, data in worker_data.items():
            dataset_size, quantized_pair, dequant = data
            parameters[worker_id] = dequant(quantized_pair)
            dataset_sizes[worker_id] = dataset_size

        weights = AggregationServer.get_dataset_ratios(dataset_sizes)
        return AggregationServer.weighted_avg(
            parameters,
            weights,
        )
