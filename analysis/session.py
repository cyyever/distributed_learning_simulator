import functools
import json
import os
import pickle

import dill


def get_worker_stat(session_dir: str) -> dict:
    worker_data: dict = {}
    for root, dirs, __ in os.walk(os.path.join(session_dir, "..")):
        for name in dirs:
            if name.startswith("worker"):
                worker_data[name] = {}
                with open(
                    os.path.join(root, name, "hyper_parameter.pk"),
                    "rb",
                ) as f:
                    worker_data[name]["hyper_parameter"] = dill.load(f)
                if os.path.isfile(os.path.join(root, name, "graph_worker_stat.json")):
                    with open(
                        os.path.join(root, name, "graph_worker_stat.json"),
                        "rt",
                        encoding="utf8",
                    ) as f:
                        worker_data[name] = json.load(f)
    return worker_data


class Session:
    def __init__(self, session_dir: str) -> None:
        assert session_dir
        with open(
            os.path.join(session_dir, "round_record.json"), "rt", encoding="utf8"
        ) as f:
            self.round_record = json.load(f)
        self.round_record = {int(k): v for k, v in self.round_record.items()}
        with open(os.path.join(session_dir, "config.pkl"), "rb") as f:
            self.config = pickle.load(f)

        self.worker_data: dict = get_worker_stat(session_dir)
        if not self.worker_data:
            raise RuntimeError(os.path.join(session_dir, ".."))

    @functools.cached_property
    def rounds(self) -> list:
        return sorted(self.round_record.keys())

    @functools.cached_property
    def last_round(self) -> int:
        return self.rounds[-1]

    @functools.cached_property
    def last_test_acc(self) -> float:
        return self.round_record[self.last_round]["test_accuracy"]

    @functools.cached_property
    def mean_test_acc(self) -> float:
        total_acc = 0
        for r in self.round_record.values():
            total_acc += r["test_accuracy"]
        return total_acc / len(self.round_record)
