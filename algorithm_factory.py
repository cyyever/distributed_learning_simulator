import functools
import itertools

from cyy_torch_toolbox.device import get_devices

from config import DistributedTrainingConfig
from practitioner import PersistentPractitioner, Practitioner
from server.afd_server import SingleModelAdaptiveFedDropoutServer
from server.fed_avg_server import FedAVGServer
from server.fed_dropout_avg_server import FedDropoutAvgServer
from server.fed_obd_server import FedOBDServer
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
from worker.fed_obd_worker import FedOBDWorker
from worker.qsgd_worker import QSGDWorker
from worker.sign_sgd_worker import SignSGDWorker

cs_constructors: dict[str, tuple] = {
    "sign_SGD": (SignSGDWorker, SignSGDServer),
    "QSGD": (QSGDWorker, QSGDServer),
    "fed_avg": (FedAVGWorker, FedAVGServer),
    "fed_obd": (
        FedOBDWorker,
        FedOBDServer,
        NNADQClientEndpoint,
        NNADQServerEndpoint,
    ),
    "fed_obd_sq": (
        FedOBDWorker,
        FedOBDServer,
        StochasticQuantClientEndpoint,
        StochasticQuantServerEndpoint,
    ),
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


def get_worker_config(
    config: DistributedTrainingConfig, practitioner_ids: None | set = None
) -> dict:
    if config.parallel_number is None:
        config.parallel_number = len(get_devices())

    practitioners = []
    if practitioner_ids is None:
        data_splitter = DataSplitter(config)
        for practitioner_id in range(config.worker_number):
            practitioner = Practitioner(practitioner_id=practitioner_id)
            practitioner.add_dataset_collection(
                name=config.dc_config.dataset_name,
                indices=data_splitter.get_dataset_indices(worker_id=practitioner_id),
            )
            practitioners.append(practitioner)
    else:
        for practitioner_id in sorted(practitioner_ids):
            practitioner = PersistentPractitioner(practitioner_id=practitioner_id)
            assert config.dc_config.dataset_name in practitioner.datasets
            practitioners.append(practitioner)
        config.worker_number = len(practitioners)

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
    worker_endpoint_cls = functools.partial(
        worker_endpoint_cls,
        **config.endpoint_kwargs.get("worker", {}),
    )
    for (worker_id, practitioner), next_process_idx in zip(
        enumerate(practitioners),
        itertools.cycle(list(range(config.parallel_number))),
    ):
        if next_process_idx not in worker_map:
            worker_map[next_process_idx] = []
        worker_map[next_process_idx].append(
            {
                "practitioner": practitioner,
                "worker_constructor": worker_constructor,
                "worker_id": worker_id,
                "endpoint_cls": worker_endpoint_cls,
                "config": config,
            }
        )
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
