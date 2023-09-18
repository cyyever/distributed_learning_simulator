import os
import sys

import pandas as pd

currentdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(currentdir, ".."))

from analysis.session import Session

if __name__ == "__main__":
    output_file = "exp.txt"
    session_path = os.getenv("session_path").strip()
    assert session_path
    session = Session(session_path)
    config = session.config
    res = {}
    res["dataset_name"] = config.dc_config.dataset_name
    res["model_name"] = config.model_config.model_name
    res["round"] = config.round
    res["worker_number"] = config.worker_number
    res["distributed_algorithm"] = config.distributed_algorithm
    if config.algorithm_kwargs:
        res |= config.algorithm_kwargs
    res["last_test_acc"] = session.last_test_acc
    res["mean_test_acc"] = session.mean_test_acc
    df = pd.DataFrame([res])
    if os.path.isfile(output_file):
        old_df = pd.read_csv(output_file)
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_csv(output_file, index=False)
