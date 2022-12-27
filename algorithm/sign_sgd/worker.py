""" signSGD: Compressed Optimisation for Non-Convex Problems https://arxiv.org/abs/1802.04434 """
from worker.gradient_worker import GradientWorker


class SignSGDWorker(GradientWorker):
    def _process_gradient(self, gradient):
        return gradient.sign()
