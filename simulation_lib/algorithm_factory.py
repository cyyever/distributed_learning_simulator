import functools

from cyy_naive_lib.log import get_logger
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
    worker_number_per_process = config.get_worker_number_per_process()
    get_logger().warning(
        "There are %s workers in total, and %s workers form a group",
        len(practitioners),
        worker_number_per_process,
    )
    client_config: list[list[dict]] = []
    tmp = list(practitioners)
    while tmp:
        batch = tmp[:worker_number_per_process]
        tmp = tmp[worker_number_per_process:]
        client_config.append(
            [
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
                for practitioner in batch
            ]
        )
    assert client_config
    result["worker"] = client_config
    return result
