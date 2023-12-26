import functools
import itertools

from cyy_naive_lib.topology.central_topology import ProcessPipeCentralTopology
from cyy_torch_toolbox.data_structure.torch_process_context import \
    TorchProcessContext

from .config import DistributedTrainingConfig
from .method.algorithm_factory import CentralizedAlgorithmFactory


def get_worker_config(
    config: DistributedTrainingConfig, practitioners: None | set = None
) -> dict:
    if practitioners is None:
        practitioners = config.create_practitioners()
    else:
        config.worker_number = len(practitioners)
        for worker_id, practitioner in enumerate(
            sorted(practitioners, key=lambda p: p.id)
        ):
            assert practitioner.has_dataset(config.dc_config.dataset_name)
            practitioner.set_worker_id(worker_id)
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
    for practitioner, next_process_idx in zip(
        practitioners, itertools.cycle(list(range(config.parallel_number)))
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
                        "worker_id": practitioner.worker_id,
                    },
                    kwargs={
                        "config": config,
                        "practitioner": practitioner,
                    },
                ),
            }
        )
    assert client_config
    result["worker"] = client_config
    return result
