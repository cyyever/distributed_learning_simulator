import functools
import json
import os
import pickle


class Session:
    def __init__(self, session_dir: str):
        with open(
            os.path.join(session_dir, "round_record.json"), "rt", encoding="utf8"
        ) as f:
            self.round_record = json.load(f)
        self.round_record = {int(k): v for k, v in self.round_record.items()}
        with open(os.path.join(session_dir, "config.pkl"), "rb") as f:
            self.config = pickle.load(f)

    @functools.cached_property
    def rounds(self) -> list:
        return sorted(self.round_record.keys())

    @functools.cached_property
    def last_round(self) -> int:
        return self.rounds[-1]

    @functools.cached_property
    def last_test_acc(self) -> float:
        return self.round_record[self.last_round]["test_acc"]

    @functools.cached_property
    def mean_test_acc(self) -> float:
        total_acc = 0
        for r in self.round_record.values():
            total_acc += r["test_acc"]
        return total_acc / len(self.round_record)
