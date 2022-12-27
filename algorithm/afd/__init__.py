from _algorithm_factory import CentralizedAlgorithmFactory

from .server import SingleModelAdaptiveFedDropoutServer
from .worker import SingleModelAdaptiveFedDropoutWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="single_model_afd",
    client_cls=SingleModelAdaptiveFedDropoutWorker,
    server_cls=SingleModelAdaptiveFedDropoutServer,
)
