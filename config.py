import argparse
import datetime
import os
import typing
import uuid

from cyy_torch_toolbox.default_config import DefaultConfig


class ExperimentConfig(DefaultConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distributed_algorithm: str = None
        self.worker_number = None
        self.parallel_number = None
        self.round = None
        self.iid = True
        self.no_distribute_init_parameters = False
        self.noise_percents: typing.Optional[list] = None
        self.__task_time = None
        self.task_id = None
        self.log_file = None

    def load_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument("--distributed_algorithm", type=str, required=True)
        parser.add_argument("--worker_number", type=int, required=True)
        parser.add_argument("--parallel_number", type=int, default=None)
        parser.add_argument("--round", type=int, required=True)
        parser.add_argument("--noniid", action="store_true", default=False)
        parser.add_argument(
            "--no_distribute_init_parameters", action="store_true", default=False
        )
        parser.add_argument("--noise_percents", type=str, default=None)
        args = super().load_args(parser=parser)

        if args.noniid:
            self.iid = False
            if self.noise_percents is not None:
                self.noise_percents = [float(s) for s in self.noise_percents.split("|")]
                assert len(self.noise_percents) == self.worker_number
        else:
            assert self.noise_percents is None
        self.__task_time = datetime.datetime.now()
        date_time = "{date:%Y-%m-%d_%H_%M_%S}".format(date=self.__task_time)
        dir_suffix = os.path.join(
            self.distributed_algorithm + ""
            if self.iid
            else self.distributed_algorithm + "_non_iid",
            self.dc_config.dataset_name,
            self.model_name,
            date_time,
            str(uuid.uuid4()),
        )
        self.save_dir = os.path.join("session", dir_suffix)
        self.task_id = "{}_{}_{}_{}".format(
            self.distributed_algorithm
            if self.iid
            else self.distributed_algorithm + "_non_iid",
            self.dc_config.dataset_name,
            self.model_name,
            date_time,
        )

        self.log_file = str(os.path.join("log", dir_suffix)) + ".log"
        return self


def get_config(parser: argparse.ArgumentParser | None = None) -> ExperimentConfig:
    config = ExperimentConfig()
    return config.load_args(parser=parser)
