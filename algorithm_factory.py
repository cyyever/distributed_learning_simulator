import functools
import itertools

from cyy_naive_lib.topology.central_topology import ProcessPipeCentralTopology
from cyy_torch_toolbox.data_structure.torch_process_context import \
    TorchProcessContext
from cyy_torch_toolbox.dataset import get_dataset_collection_sampler

from _algorithm_factory import CentralizedAlgorithmFactory
from algorithm import register_algorithms
from config import DistributedTrainingConfig
from practitioner import PersistentPractitioner, Practitioner

register_algorithms()


def get_worker_config(
    config: DistributedTrainingConfig, practitioner_ids: None | set = None
) -> dict:
    practitioners = []
    if practitioner_ids is None:
        sampler = get_dataset_collection_sampler(
            name=config.dataset_sampling,
            dataset_collection=config.create_dataset_collection(),
            part_number=config.worker_number,
            **config.dataset_sampling_kwargs
        )
        for practitioner_id in range(config.worker_number):
            practitioner = Practitioner(
                practitioner_id=practitioner_id, worker_id=practitioner_id
            )
            practitioner.set_sampler(
                name=config.dc_config.dataset_name, sampler=sampler
            )
            practitioners.append(practitioner)
    else:
        for worker_id, practitioner_id in enumerate(sorted(practitioner_ids)):
            practitioner = PersistentPractitioner(
                practitioner_id=practitioner_id, worker_id=worker_id
            )
            assert practitioner.has_dataset(config.dc_config.dataset_name)
            practitioners.append(practitioner)
        config.worker_number = len(practitioners)

    assert practitioners
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
    assert client_config
    result["worker"] = client_config
    return result
