import functools
import itertools

from cyy_naive_lib.topology.central_topology import ProcessPipeCentralTopology
from cyy_torch_toolbox.data_structure.torch_process_context import \
    TorchProcessContext

from _algorithm_factory import CentralizedAlgorithmFactory
from algorithm import register_algorithms
from config import DistributedTrainingConfig
from practitioner import PersistentPractitioner, Practitioner
from sampler import get_dataset_sampler

register_algorithms()


def get_worker_config(
    config: DistributedTrainingConfig, practitioner_ids: None | set = None
) -> dict:
    practitioners = []
    if practitioner_ids is None:
        sampler = get_dataset_sampler(config)
        for practitioner_id in range(config.worker_number):
            practitioner = Practitioner(practitioner_id=practitioner_id)
            practitioner.add_dataset_collection(
                name=config.dc_config.dataset_name,
                indices=sampler.get_dataset_indices(worker_id=practitioner_id),
            )
            practitioners.append(practitioner)
    else:
        for practitioner_id in sorted(practitioner_ids):
            practitioner = PersistentPractitioner(practitioner_id=practitioner_id)
            assert practitioner.has_dataset(config.dc_config.dataset_name)
            practitioners.append(practitioner)
        config.worker_number = len(practitioners)

    assert CentralizedAlgorithmFactory.has_algorithm(config.distributed_algorithm)
    topology = ProcessPipeCentralTopology(
        mp_context=TorchProcessContext(), worker_num=config.worker_number
    )
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
                "constructor": functools.partial(
                    CentralizedAlgorithmFactory.create_client,
                    algorithm_name=config.distributed_algorithm,
                    endpoint_kwargs=config.endpoint_kwargs.get("worker", {})
                    | {
                        "worker_id": worker_id,
                    },
                    kwargs={
                        "config": config,
                        "practitioner": practitioner,
                        "worker_id": worker_id,
                    },
                ),
            }
        )
    result["worker"] = client_config
    return result
