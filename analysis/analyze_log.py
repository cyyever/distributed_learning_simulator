import os
import re
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(currentdir, ".."))

import torch
from config import global_config as config
from config import load_config


def compute_acc(paths: list) -> None:
    final_test_acc = []
    worker_acc: dict = {}
    for path in paths:
        assert os.path.isfile(path)
        lines = None
        with open(path, "rt", encoding="utf8") as f:
            lines = f.readlines()
        for line in reversed(lines):
            if config.distributed_algorithm == "sign_SGD":
                if "test loss" in line:
                    res = re.findall("[0-9.]+%", line)
                    assert len(res) == 1
                    acc = float(res[0].replace("%", ""))
                    final_test_acc.append(acc)
                    break
            elif config.distributed_algorithm in (
                "fed_obd_first_stage",
                "fed_obd_layer",
            ):
                if (
                    "test in" in line
                    and "accuracy" in line
                    and f"round: {config.round}" in line
                ):
                    print("line is", line)
                    res = re.findall("[0-9.]+%", line)
                    assert len(res) == 1
                    acc = float(res[0].replace("%", ""))
                    final_test_acc.append(acc)
                    break
            else:
                if "test in" in line and "accuracy" in line:
                    res = re.findall("[0-9.]+%", line)
                    assert len(res) == 1
                    acc = float(res[0].replace("%", ""))
                    print(line)
                    final_test_acc.append(acc)
                    break
        for worker_id in range(config.worker_number):
            for line in reversed(lines):
                res = re.findall(f"worker {worker_id}.*train.*accuracy", line)
                if res:
                    res = re.findall("[0-9.]+%", line)
                    assert len(res) == 1
                    acc = float(res[0].replace("%", ""))
                    if worker_id not in worker_acc:
                        worker_acc[worker_id] = []
                    worker_acc[worker_id].append(acc)
                    break
    assert len(final_test_acc) == len(paths)
    std, mean = torch.std_mean(torch.tensor(final_test_acc))
    print("test acc", round(mean.item(), 2), round(std.item(), 2))


def compute_data_amount(paths: list) -> dict:
    trainer = config.create_trainer()
    parameter_list = trainer.model_util.get_parameter_list()
    distributed_algorithm = config.distributed_algorithm.lower()
    print("worker_number is", config.worker_number)
    print("model is", config.model_config.model_name)
    uploaded_msg_num = config.round * config.algorithm_kwargs.get(
        "random_client_number", config.worker_number
    )
    uploaded_parameter_num = uploaded_msg_num * parameter_list.nelement()
    downloaded_msg_num = uploaded_msg_num
    downloaded_parameter_num = uploaded_parameter_num
    distributed_msg_num = config.worker_number
    distributed_parameter_num = distributed_msg_num * parameter_list.nelement()
    msg_num = uploaded_msg_num + downloaded_msg_num + distributed_msg_num
    data_amount: float | tuple = 0
    match distributed_algorithm:
        case "fed_avg":
            data_amount = (
                parameter_list.nelement()
                * parameter_list.element_size()
                * msg_num
                / (1024 * 1024)
            )
        case "fed_obd":
            msg_num += (
                config.algorithm_kwargs["second_phase_epoch"] * config.worker_number * 2
            )

            data_amounts = []
            for path in paths:
                remain_msg = msg_num
                lines = None
                compressed_part = 0
                rnd_cnt = 0
                with open(path, "rt", encoding="utf8") as f:
                    lines = f.readlines()
                stage_one = True
                for line in lines:
                    if "broadcast NNABQ compression ratio" in line:
                        res = re.findall("[0-9.]+$", line)
                        assert len(res) == 1
                        broadcast_ratio = float(
                            res[0].replace("(", "").replace(",", "")
                        )
                        # print("broadcast_ratio", broadcast_ratio)
                        rnd_cnt += 1
                        if rnd_cnt <= config.round:
                            compressed_part += (
                                broadcast_ratio
                                * config.algorithm_kwargs["random_client_number"]
                            )
                            remain_msg -= config.algorithm_kwargs[
                                "random_client_number"
                            ]
                        else:
                            stage_one = False
                            if remain_msg > config.worker_number:
                                compressed_part += (
                                    broadcast_ratio * config.worker_number
                                )
                                remain_msg -= config.worker_number
                    if "worker NNABQ compression ratio" in line:
                        res = re.findall("[0-9.]+$", line)
                        assert len(res) == 1
                        worker_ratio = float(res[0].replace("(", "").replace(",", ""))
                        # print("worker_ratio is ", worker_ratio)
                        if stage_one:
                            worker_ratio *= 1 - config.algorithm_kwargs["dropout_rate"]
                        compressed_part += worker_ratio
                        remain_msg -= 1
                # assert remain_msg == 0
                print(remain_msg)
                assert remain_msg == config.worker_number
                compressed_part += remain_msg
                data_amounts.append(
                    parameter_list.nelement()
                    * parameter_list.element_size()
                    * compressed_part
                    / (1024 * 1024)
                )
            assert len(data_amounts) == len(paths)
            std, mean = torch.std_mean(torch.tensor(data_amounts))
            data_amount = {"mean": round(mean.item(), 2), "std": round(std.item(), 2)}
        case "fed_obd_sq":
            msg_num += (
                config.algorithm_kwargs["second_phase_epoch"] * config.worker_number * 2
            )

            data_amount = (
                uploaded_parameter_num * (1 - config.algorithm_kwargs["dropout_rate"])
                + downloaded_parameter_num
                + config.algorithm_kwargs["second_phase_epoch"]
                * config.worker_number
                * 2
                * parameter_list.nelement()
                + distributed_parameter_num * parameter_list.element_size()
            ) / (1024 * 1024)
        case "fed_dropout_avg":
            data_amounts = []
            for path in paths:
                lines = None
                with open(path, "rt", encoding="utf8") as f:
                    lines = f.readlines()
                uploaded_parameter_num = 0
                for line in lines:
                    if "send_num" in line:
                        res = re.findall("[0-9.]+$", line)
                        assert len(res) == 1
                        uploaded_parameter_num += float(res[0])
                assert uploaded_parameter_num > 0
                assert downloaded_parameter_num > 0
                data_amounts.append(
                    (
                        uploaded_parameter_num
                        + (downloaded_parameter_num + distributed_parameter_num)
                    )
                    * parameter_list.element_size()
                    / (1024 * 1024)
                )
            std, mean = torch.std_mean(torch.tensor(data_amounts))
            data_amount = {"mean": round(mean.item(), 2), "std": round(std.item(), 2)}
        case "single_model_afd":
            data_amounts = []
            for path in paths:
                transfer_number = 0
                lines = None
                with open(path, "rt", encoding="utf8") as f:
                    lines = f.readlines()
                for line in lines:
                    if "send_num" in line:
                        res = re.findall("[0-9.]+$", line)
                        assert len(res) == 1
                        transfer_number += float(res[0])
                data_amounts.append(
                    (transfer_number + distributed_parameter_num)
                    * parameter_list.element_size()
                    / (1024 * 1024)
                )
            std, mean = torch.std_mean(torch.tensor(data_amounts))
            data_amount = {"mean": round(mean.item(), 2), "std": round(std.item(), 2)}

        case "fed_obd_first_stage":
            data_amounts = []
            for path in paths:
                remain_msg = msg_num
                lines = None
                compressed_part = 0
                rnd_cnt = 0
                with open(path, "rt", encoding="utf8") as f:
                    lines = f.readlines()
                stage_one = True
                for line in lines:
                    if "broadcast NNABQ compression ratio" in line:
                        res = re.findall("[0-9.]+$", line)
                        assert len(res) == 1
                        broadcast_ratio = float(
                            res[0].replace("(", "").replace(",", "")
                        )
                        # print("broadcast_ratio", broadcast_ratio)
                        rnd_cnt += 1
                        if rnd_cnt <= config.round:
                            compressed_part += (
                                broadcast_ratio
                                * config.algorithm_kwargs["random_client_number"]
                            )
                            remain_msg -= config.algorithm_kwargs[
                                "random_client_number"
                            ]
                        else:
                            break
                    if "worker NNABQ compression ratio" in line:
                        res = re.findall("[0-9.]+$", line)
                        assert len(res) == 1
                        worker_ratio = float(res[0].replace("(", "").replace(",", ""))
                        # print("worker_ratio is ", worker_ratio)
                        if stage_one:
                            worker_ratio *= 1 - config.algorithm_kwargs["dropout_rate"]
                        compressed_part += worker_ratio
                        remain_msg -= 1
                # assert remain_msg == 0
                print(remain_msg)
                # assert remain_msg == config.worker_number
                compressed_part += config.worker_number
                data_amounts.append(
                    parameter_list.nelement()
                    * parameter_list.element_size()
                    * compressed_part
                    / (1024 * 1024)
                )
            assert len(data_amounts) == len(paths)
            std, mean = torch.std_mean(torch.tensor(data_amounts))
            data_amount = {"mean": round(mean.item(), 2), "std": round(std.item(), 2)}

        case "fed_paq":
            msg_num = (
                config.round * config.algorithm_kwargs["random_client_number"] * 2
                + config.worker_number
            )
            data_amount = (
                uploaded_parameter_num * 1
                + downloaded_parameter_num * parameter_list.element_size()
                + distributed_parameter_num * parameter_list.element_size()
            ) / (1024 * 1024)
    match data_amount:
        case float():
            data_amount = round(data_amount, 2)
        case dict():
            data_amount = {k: round(v, 2) for k, v in data_amount.items()}

    return {"msg_num": msg_num, "data_amount": data_amount}


if __name__ == "__main__":
    load_config()
    # config.distributed_algorithm = "fed_obd_first_stage"
    paths = os.getenv("logfiles").strip().split(" ")
    assert paths
    compute_acc(paths)
    res = compute_data_amount(paths)
    print("msg_num is", res["msg_num"])
    print("data_amount is", res["data_amount"])
