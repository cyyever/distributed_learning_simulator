""" signSGD: Compressed Optimisation for Non-Convex Problems https://arxiv.org/abs/1802.04434 """
from .gradient_worker import GradientWorker


class SignSGDWorker(GradientWorker):
    def _process_gradient(self, gradient):
        self.send_data_to_server(
            {"dataset_size": self.trainer.dataset_size, "gradient": gradient.sign()}
        )
        return self.get_result_from_server()["gradient"]
