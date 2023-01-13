import functools
import itertools

import algorithm
from _algorithm_factory import CentralizedAlgorithmFactory
from config import DistributedTrainingConfig
from practitioner import PersistentPractitioner, Practitioner
from server.fed_avg_server import FedAVGServer
from topology.central_topology import ProcessCentralTopology
# from topology.peer_endpoint import PeerEndpoint
# from topology.peer_to_peer_topology import ProcessPeerToPeerTopology
from util.data_spliting import DataSplitter
from worker.fed_avg_worker import FedAVGWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_avg",
    client_cls=FedAVGWorker,
    server_cls=FedAVGServer,
)

# peer_constructors: dict[str, tuple] = {}
# peer_constructors["personalized_shapley_value"] = (
#     PersonalizedShapleyValueWorker,
#     QuantizedPeerEndpoint,
# )


def get_worker_config(
    config: DistributedTrainingConfig, practitioner_ids: None | set = None
) -> dict:
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

    if CentralizedAlgorithmFactory.has_algorithm(config.distributed_algorithm):
        topology = ProcessCentralTopology(worker_num=config.worker_number)
        result: dict = {"topology": topology}
        result["server"] = {}
        result["server"]["constructor"] = functools.partial(
            CentralizedAlgorithmFactory.create_server,
            algorithm_name=config.distributed_algorithm,
            endpoint_kwargs=config.endpoint_kwargs.get("server", {}),
            kwargs={"config": config},
        )
        client_config: dict = {}
        for (worker_id, practitioner), next_process_idx in zip(
            enumerate(practitioners),
            itertools.cycle(list(range(config.parallel_number))),
        ):
            if next_process_idx not in client_config:
                client_config[next_process_idx] = []
            client_config[next_process_idx].append(
                {
                    "practitioner": practitioner,
                    "constructor": functools.partial(
                        CentralizedAlgorithmFactory.create_client,
                        algorithm_name=config.distributed_algorithm,
                        endpoint_kwargs=config.endpoint_kwargs.get("worker", {})
                        | {
                            "worker_id": worker_id,
                        },
                        kwargs={
                            "config": config,
                            "worker_id": worker_id,
                        },
                    ),
                    "config": config,
                }
            )
        result["worker"] = client_config
        return result
