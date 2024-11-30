import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cyy_naive_lib.fs.path import find_directories
from simulation_lib.config import load_config_from_file


def extract_data(
    session_path: str, algorithm: str, aggregated_performance_metric: dict
) -> dict:
    for server_session in find_directories(session_path, "visualizer"):
        if "server" not in server_session:
            continue
        for round_no in range(1, config.round + 1):
            round_str = f"round_{round_no}"

            metric_file = os.path.join(
                server_session, round_str, "test", "performance_metric.json"
            )
            if not os.path.isfile(metric_file):
                continue
            with open(
                metric_file,
                encoding="utf8",
            ) as f:
                performance_metric = json.load(f)
                for k, v in performance_metric.items():
                    if k not in aggregated_performance_metric:
                        aggregated_performance_metric[k] = pd.DataFrame(
                            columns=["round", k]
                        )
                    aggregated_performance_metric[k] = pd.concat(
                        [
                            aggregated_performance_metric[k],
                            pd.DataFrame(
                                [[round_no, list(v.values())[0], algorithm]],
                                columns=["round", k, "algorithm"],
                            ),
                        ]
                    )
    return aggregated_performance_metric


if __name__ == "__main__":
    aggregated_performance_metric: dict = {}
    config_files = os.getenv("config_files")
    assert config_files is not None
    for config_file in config_files.split():
        config = load_config_from_file(config_file=config_file)
        session_path = (
            f"session/{config.distributed_algorithm}/{config.dc_config.dataset_name}/"
        )
        extract_data(
            session_path, config.distributed_algorithm, aggregated_performance_metric
        )
    for metric, metric_df in aggregated_performance_metric.items():
        print(f"deal with {metric}")
        ax = sns.lineplot(
            data=metric_df, x="round", y=metric, hue="algorithm", errorbar="sd"
        )
        plt.tight_layout()
        plt.savefig(f"{metric}.png")
        plt.clf()
