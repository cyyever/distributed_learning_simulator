from typing import Callable

from cyy_naive_lib.topology.cs_endpoint import ClientEndpoint, ServerEndpoint


class CentralizedAlgorithmFactory:
    config: dict[str, dict] = {}

    @classmethod
    def register_algorithm(
        cls,
        algorithm_name: str,
        client_cls: Callable,
        server_cls: Callable,
        client_endpoint_cls: None | Callable = None,
        server_endpoint_cls: None | Callable = None,
        algorithm_cls: None | Callable = None,
    ) -> None:
        assert algorithm_name not in cls.config
        if client_endpoint_cls is None:
            client_endpoint_cls = ClientEndpoint
        if server_endpoint_cls is None:
            server_endpoint_cls = ServerEndpoint
        cls.config[algorithm_name] = {
            "client_cls": client_cls,
            "server_cls": server_cls,
            "client_endpoint_cls": client_endpoint_cls,
            "server_endpoint_cls": server_endpoint_cls,
        }
        if algorithm_cls is not None:
            cls.config[algorithm_name]["algorithm_cls"] = algorithm_cls

    @classmethod
    def has_algorithm(cls, algorithm_name: str) -> bool:
        return algorithm_name in cls.config

    @classmethod
    def create_client(
        cls,
        algorithm_name: str,
        kwargs: dict,
        endpoint_kwargs: dict,
        extra_kwargs: dict | None = None,
        extra_endpoint_kwargs: dict | None = None,
    ) -> None:
        config = cls.config[algorithm_name]
        if extra_kwargs is None:
            extra_kwargs = {}
        if extra_endpoint_kwargs is None:
            extra_endpoint_kwargs = {}
        endpoint = config["client_endpoint_cls"](
            **(endpoint_kwargs | extra_endpoint_kwargs)
        )
        return config["client_cls"](endpoint=endpoint, **(kwargs | extra_kwargs))

    @classmethod
    def create_server(
        cls,
        algorithm_name: str,
        kwargs: dict,
        endpoint_kwargs: dict,
        extra_kwargs: dict | None = None,
        extra_endpoint_kwargs: dict | None = None,
    ) -> None:
        config = cls.config[algorithm_name]
        if extra_kwargs is None:
            extra_kwargs = {}
        if extra_endpoint_kwargs is None:
            extra_endpoint_kwargs = {}
        endpoint = config["server_endpoint_cls"](
            **(endpoint_kwargs | extra_endpoint_kwargs)
        )
        algorithm = None
        if "algorithm_cls" in config:
            algorithm = config["algorithm_cls"]()
            assert "algorithm" not in extra_kwargs
            extra_kwargs["algorithm"] = algorithm

        return config["server_cls"](endpoint=endpoint, **(kwargs | extra_kwargs))
