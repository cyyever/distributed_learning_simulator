import json
import os

import pandas as pd
import torch

from .session import Session


def dump_analysis() -> None:
    session_path = os.getenv("session_path")
    assert session_path is not None
    session_path = session_path.strip()
    session = Session(session_path)
    config = session.config
    res = {}
    res["exp_name"] = config.exp_name
    res["distributed_algorithm"] = config.distributed_algorithm
    res["dataset_name"] = config.dc_config.dataset_name
    res["model_name"] = config.model_config.model_name
    if config.endpoint_kwargs:
        res["endpoint_kwargs"] = config.endpoint_kwargs
    if config.trainer_config.dataloader_kwargs:
        res["dataloader_kwargs"] = config.trainer_config.dataloader_kwargs
    res["round"] = len(session.rounds)
    res["worker_number"] = config.worker_number
    if config.algorithm_kwargs:
        res |= config.algorithm_kwargs
    res["dataset_sampling"] = config.dataset_sampling
    if config.dataset_sampling_kwargs:
        res |= config.dataset_sampling_kwargs
    res["last_test_acc"] = session.last_test_acc
    res["mean_test_acc"] = session.mean_test_acc

    total_worker_cnts = {
        "exp_name": config.exp_name,
        "dataset_name": config.dc_config.dataset_name,
    }

    # Analysis worker data here
    for data in session.worker_data.values():
        for k, v in data.items():
            if "cnt" in k or "byte" in k:
                if k in ("embedding_bytes", "model_bytes"):
                    total_worker_cnts[k] = v
                    continue
                if "edge_cnt" in k or "node_cnt" in k:
                    if k not in total_worker_cnts:
                        total_worker_cnts[k] = []
                    total_worker_cnts[k].append(v)
                    continue
                if k not in total_worker_cnts:
                    total_worker_cnts[k] = {}
                for k2, v2 in v.items():
                    if k2 not in total_worker_cnts[k]:
                        total_worker_cnts[k][k2] = v2
                    else:
                        total_worker_cnts[k][k2] = total_worker_cnts[k][k2] + v2
    for k, v in total_worker_cnts.items():
        if "edge_cnt" in k or "node_cnt" in k:
            std, mean = torch.std_mean(torch.tensor(v, dtype=torch.float))
            total_worker_cnts[k] = {"mean": mean.item(), "std": std.item()}

    res |= total_worker_cnts
    res |= {"performance": session.round_record}
    for k, v in res.items():
        if isinstance(v, dict):
            res[k] = json.dumps(v)
    col_list = [
        "distributed_algorithm",
        "dataset_name",
        "model_name",
        "last_test_acc",
        "mean_test_acc",
        "round",
        "worker_number",
    ]
    if config.exp_name:
        col_list = ["exp_name"] + col_list
    for k in config.algorithm_kwargs:
        col_list.append(k)
    col_list += list(set(res.keys()) - set(col_list))
    df = pd.DataFrame([res])
    df = df[col_list]
    # df = df.drop_duplicates(ignore_index=True)
    # df = df.sort_values(by=col_list, ascending=False, ignore_index=True)
    output_file = "exp.txt"
    if os.path.isfile(output_file):
        old_df = pd.read_csv(output_file)
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_csv(output_file, index=False)
    df.to_excel("exp.xlsx", index=False, sheet_name="result")
    df.to_json("exp.json")
