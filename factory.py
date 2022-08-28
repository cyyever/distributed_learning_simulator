import functools
from typing import Any

from cyy_torch_toolbox.device import get_devices

from server.afd_server import SingleModelAdaptiveFedDropoutServer
from server.fed_avg_server import FedAVGServer
from server.fed_dropout_avg_server import FedDropoutAvgServer
from server.GTG_shapley_value_server import GTGShapleyValueServer
from server.multiround_shapley_value_server import MultiRoundShapleyValueServer
from server.qsgd_server import QSGDServer
from server.sign_sgd_server import SignSGDServer
from topology.central_topology import ProcessCentralTopology
from topology.cs_endpoint import ClientEndpoint, ServerEndpoint
from topology.peer_endpoint import PeerEndpoint
from topology.peer_to_peer_topology import ProcessPeerToPeerTopology
from topology.quantized_endpoint import (NNADQClientEndpoint,
                                         NNADQServerEndpoint,
                                         StochasticQuantClientEndpoint,
                                         StochasticQuantServerEndpoint)
from util.data_spliting import DataSplitter
from worker.afd_worker import SingleModelAdaptiveFedDropoutWorker
from worker.fed_avg_worker import FedAVGWorker
from worker.fed_dropout_avg_worker import FedDropoutAvgWorker
from worker.qsgd_worker import QSGDWorker
from worker.sign_sgd_worker import SignSGDWorker

cs_constructors: dict[str, tuple] = {
    "sign_SGD": (SignSGDWorker, SignSGDServer),
    "QSGD": (QSGDWorker, QSGDServer),
    "fed_avg": (FedAVGWorker, FedAVGServer),
    "single_model_afd": (
        SingleModelAdaptiveFedDropoutWorker,
        SingleModelAdaptiveFedDropoutServer,
    ),
    "fed_dropout_avg": (
        FedDropoutAvgWorker,
        FedDropoutAvgServer,
    ),
    "fed_paq": (
        FedAVGWorker,
        FedAVGServer,
        StochasticQuantClientEndpoint,
        StochasticQuantServerEndpoint,
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
# peer_constructors["personalized_shapley_value"] = (
#     PersonalizedShapleyValueWorker,
#     QuantizedPeerEndpoint,
# )


def get_worker_config(config: Any) -> dict:
    if config.parallel_number is None:
        config.parallel_number = len(get_devices())

    data_splitter = DataSplitter(config)

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

    worker_map: dict = {}
    next_process_idx = 0
    for worker_id in range(config.worker_number):
        if next_process_idx not in worker_map:
            worker_map[next_process_idx] = []
        endpoint_cls = functools.partial(
            worker_endpoint_cls,
            worker_id=worker_id,
            **config.endpoint_kwargs.get("worker", {}),
        )
        worker_map[next_process_idx].append(
            {
                "worker_id": worker_id,
                "worker_constructor": worker_constructor,
                "endpoint_cls": endpoint_cls,
                "data_splitter": data_splitter,
                "config": config,
            }
        )
        next_process_idx = (next_process_idx + 1) % config.parallel_number
    result: dict = {
        "worker_map": worker_map,
    }
    result["topology"] = topology
    if server_constructor is not None:
        result["server_config"] = {}
        result["server_config"]["server_constructor"] = functools.partial(
            server_constructor,
            config=config,
        )
        result["server_config"]["server_endpoint_cls"] = functools.partial(
            server_endpoint_cls, **config.endpoint_kwargs.get("server", {})
        )
    return result
