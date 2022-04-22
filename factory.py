import functools

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.device import get_devices
from cyy_torch_toolbox.ml_type import MachineLearningPhase

from server.fed_avg_server import FedAVGServer
from topology.central_topology import ProcessCentralTopology
from topology.cs_endpoint import (ClientEndpoint, QuantizedClientEndpoint,
                                  QuantizedServerEndpoint, ServerEndpoint)
from topology.endpoint import Endpoint
from topology.peer_endpoint import PeerEndpoint, QuantizedPeerEndpoint
from topology.peer_to_peer_topology import ProcessPeerToPeerTopology

try:
    from server.GTG_shapley_value_server import GTGShapleyValueServer
    from server.multiround_shapley_value_server import \
        MultiRoundShapleyValueServer
    from worker.personalized_shapley_value_worker import \
        PersonalizedShapleyValueWorker

    has_private_algorithm = True
except BaseException:
    has_private_algorithm = False
from server.qsgd_server import QSGDServer
from server.sign_sgd_server import SignSGDServer
from worker.fed_avg_worker import FedAVGWorker
from worker.fed_one_worker import FedOneWorker
from worker.qsgd_worker import QSGDWorker
from worker.sign_sgd_worker import SignSGDWorker

cs_constructors: dict[str, tuple] = {
    "sign_SGD": (SignSGDWorker, SignSGDServer),
    "QSGD": (QSGDWorker, QSGDServer),
    "fed_avg": (FedAVGWorker, FedAVGServer),
    "fed_avg_one": (FedOneWorker, FedAVGServer),
    "fed_paq": (
        FedAVGWorker,
        FedAVGServer,
        QuantizedServerEndpoint,
        QuantizedClientEndpoint,
    ),
    "multiround_shapley_value": (
        FedAVGWorker,
        MultiRoundShapleyValueServer,
    ),
    "GTG_shapley_value": (
        FedAVGWorker,
        GTGShapleyValueServer,
    ),
}

peer_constructors: dict[str, tuple] = {}


def get_worker_config(config) -> dict:
    if config.parallel_number is None:
        config.parallel_number = len(get_devices())

    dc = config.create_dataset_collection()
    if config.iid or config.noise_percents is not None:
        training_dataset_indices = dc.get_dataset_util(
            MachineLearningPhase.Training
        ).iid_split_indices([1] * config.worker_number)
    else:
        get_logger().warning("use non IID training dataset")
        training_dataset_indices = dc.get_dataset_util(
            MachineLearningPhase.Training
        ).random_split_indices([1] * config.worker_number)
    del dc
    cs_constructor = cs_constructors.get(config.distributed_algorithm, None)
    if cs_constructor is not None:
        (
            worker_constructor,
            server_constructor,
            worker_endpoint_cls,
            server_endpoint_cls,
            *_,
        ) = (*cs_constructor, ClientEndpoint, ServerEndpoint, None)
    else:
        peer_constructor = peer_constructors.get(config.distributed_algorithm, None)
        if peer_constructor is not None:
            worker_constructor, worker_endpoint_cls, *_ = (
                *peer_constructor,
                PeerEndpoint,
                None,
            )
        else:
            raise RuntimeError(f"unknown algorithm {config.distributed_algorithm}")
    if server_constructor is not None:
        topology = ProcessCentralTopology(worker_num=config.worker_number)
    else:
        topology = ProcessPeerToPeerTopology(worker_num=config.worker_number)
    topology.create_global_lock()

    worker_map: dict = {}
    next_process_idx = 0
    for worker_id in range(config.worker_number):
        if next_process_idx not in worker_map:
            worker_map[next_process_idx] = []
        endpoint: Endpoint = worker_endpoint_cls(topology=topology, worker_id=worker_id)
        worker_map[next_process_idx].append(
            {
                "worker_id": worker_id,
                "worker_constructor": functools.partial(
                    worker_constructor, endpoint=endpoint
                ),
                "training_dataset_indices": training_dataset_indices[worker_id],
                "config": config,
                "noise_percent": config.noise_percents[worker_id]
                if config.noise_percents is not None
                else None,
            }
        )
        next_process_idx = (next_process_idx + 1) % config.parallel_number
    result: dict = {
        "worker_map": worker_map,
    }
    if server_constructor is not None:
        result["server_constructor"] = functools.partial(
            server_constructor,
            endpoint=server_endpoint_cls(topology),
            config=config,
        )
    return result
