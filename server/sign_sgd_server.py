""" signSGD: Compressed Optimisation for Non-Convex Problems https://arxiv.org/abs/1802.04434 """

from .aggregation_server import AggregationServer


class SignSGDServer(AggregationServer):
    def _aggregate_worker_data(self,round_number, worker_data):
        gradients = {}
        dataset_sizes = {}
        for worker_id, data in worker_data.items():
            dataset_size, signed_gradient = data
            gradients[worker_id] = signed_gradient
            dataset_sizes[worker_id] = dataset_size

        weights = AggregationServer.get_dataset_ratios(dataset_sizes)
        return AggregationServer.weighted_avg(
            gradients,
            weights,
        ).sign()
