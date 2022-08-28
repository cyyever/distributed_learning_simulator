import argparse
import os
import re

import torch


def compute_acc(paths, args):
    final_test_acc = []
    worker_acc: dict = {}
    for path in paths:
        assert os.path.isfile(path)
        lines = None
        with open(path, "rt", encoding="utf8") as f:
            lines = f.readlines()
        for line in reversed(lines):
            if args.a == "sign_SGD":
                if "test loss" in line:
                    res = re.findall("[0-9.]+%", line)
                    assert len(res) == 1
                    acc = float(res[0].replace("%", ""))
                    final_test_acc.append(acc)
                    break
            elif args.a == "fed_obd_first_stage":
                if "test accuracy is" in line and "round " + str(args.r) in line:
                    res = re.findall("[0-9.]+%", line)
                    assert len(res) == 1
                    acc = float(res[0].replace("%", ""))
                    # print(line)
                    final_test_acc.append(acc)
                    break
            else:
                if "test accuracy is" in line:
                    res = re.findall("[0-9.]+%", line)
                    assert len(res) == 1
                    acc = float(res[0].replace("%", ""))
                    print(line)
                    final_test_acc.append(acc)
                    break
        for worker_id in range(args.w):
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
    assert len(worker_acc) == args.w
    for worker_id in range(args.w):
        assert len(worker_acc[worker_id]) == len(paths)
    worker_std, worker_mean = torch.std_mean(
        torch.tensor(sum(worker_acc.values(), start=[]))
    )
    print("workers training", worker_mean, worker_std)
    std, mean = torch.std_mean(torch.tensor(final_test_acc))
    print("test acc", mean, std)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", type=str, required=True, help="algorithm")
    parser.add_argument("-w", type=int, required=True, help="worker number")
    parser.add_argument(
        "-k", type=int, required=False, default=None, help="worker subset number"
    )
    parser.add_argument("-r", type=int, required=True, help="round number")
    parser.add_argument(
        "-sl", type=int, default=None, help="local epoch of second stage"
    )
    parser.add_argument("-d", type=float, help="droupout rate")
    parser.add_argument("-f", type=str, required=True, help="logfiles")
    args = parser.parse_args()
    paths = args.f.split(" ")

    compute_acc(paths, args)

    if args.a.lower() == "fed_obd":
        assert args.k is not None
        total_msg = args.r * args.k * 2 + args.sl * args.w * 2 + args.w
        print("total_msg is", total_msg)

        avg_compression = []
        for path in paths:
            remain_msg = total_msg
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
                    broadcast_ratio = float(res[0].replace("(", "").replace(",", ""))
                    rnd_cnt += 1
                    if rnd_cnt <= args.r:
                        compressed_part += broadcast_ratio * args.k
                        remain_msg -= args.k
                    else:
                        stage_one = False
                        compressed_part += broadcast_ratio * args.w
                        remain_msg -= args.w
                if "worker NNABQ compression ratio" in line:
                    res = re.findall("[0-9.]+$", line)
                    assert len(res) == 1
                    worker_ratio = float(res[0].replace("(", "").replace(",", ""))
                    if stage_one:
                        worker_ratio *= 1 - args.d
                    compressed_part += worker_ratio
                    remain_msg -= 1
            print("remain_msg is", remain_msg, "path is", path)
            assert remain_msg == args.w
            compressed_part += remain_msg
            avg_compression.append(compressed_part / total_msg)
        assert len(avg_compression) == len(paths)
        std, mean = torch.std_mean(torch.tensor(avg_compression))
        print("compression", mean, std)
    elif args.a.lower() == "fed_obd_sq":
        total_msg = args.r * args.k * 2 + args.sl * args.w * 2 + args.w
        print("total_msg is", total_msg)

        compression = (
            args.r * args.k * (1 - args.d) / 4
            + args.r * args.k / 4
            + args.sl * args.w * 2 / 4
            + args.w
        ) / total_msg

        print("compression", compression)
        print("co",total_msg * compression)
    if args.a.lower() == "fed_dropout_avg":
        total_msg = args.r * args.k * 2 + args.w
        print("total_msg is", total_msg)

        total_number = 0
        transfer_number: float = 0
        parameter_number = None
        avg_compression = []
        for path in paths:
            lines = None
            with open(path, "rt", encoding="utf8") as f:
                lines = f.readlines()
            for line in lines:
                if "send_num" in line:
                    res = re.findall("[0-9.]+$", line)
                    assert len(res) == 1
                    transfer_number += float(res[0])
                if "total_num" in line:
                    res = re.findall("[0-9.]+$", line)
                    assert len(res) == 1
                    parameter_number = float(res[0])
                    total_number += float(res[0])
            avg_compression.append(
                (transfer_number + args.w * parameter_number)
                / (total_number + args.w * parameter_number)
            )
        std, mean = torch.std_mean(torch.tensor(avg_compression))
        print("compression", mean, std)
    if args.a.lower() == "afd":
        total_msg = args.r * args.k * 2 + args.w
        print("total_msg is", total_msg)

        total_number = 0
        transfer_number = 0
        parameter_number = None
        avg_compression = []
        for path in paths:
            lines = None
            with open(path, "rt", encoding="utf8") as f:
                lines = f.readlines()
            for line in lines:
                if "send_num" in line:
                    res = re.findall("[0-9.]+$", line)
                    assert len(res) == 1
                    transfer_number += float(res[0])
                    assert parameter_number is not None
                    total_number += parameter_number
                if "parameter number is" in line:
                    res = re.findall("[0-9.]+$", line)
                    assert len(res) == 1
                    parameter_number = float(res[0])
            avg_compression.append(
                (transfer_number + args.w * parameter_number)
                / (total_number + args.w * parameter_number)
            )
        std, mean = torch.std_mean(torch.tensor(avg_compression))
        print("compression", mean, std)

    if args.a.lower() == "fed_obd_first_stage":
        assert args.k is not None
        total_msg = args.r * args.k * 2 + args.w
        print("total_msg is", total_msg)

        avg_compression = []
        for path in paths:
            remain_msg = total_msg
            lines = None
            compressed_part = 0
            rnd_cnt = 0
            with open(path, "rt", encoding="utf8") as f:
                lines = f.readlines()
            for line in lines:
                if "switch" in line:
                    break
                if "broadcast NNABQ compression ratio" in line:
                    res = re.findall("[0-9.]+$", line)
                    assert len(res) == 1
                    broadcast_ratio = float(res[0].replace("(", "").replace(",", ""))
                    rnd_cnt += 1
                    if rnd_cnt <= args.r:
                        compressed_part += broadcast_ratio * args.k
                        remain_msg -= args.k
                if "worker NNABQ compression ratio" in line:
                    res = re.findall("[0-9.]+$", line)
                    assert len(res) == 1
                    worker_ratio = float(res[0].replace("(", "").replace(",", ""))
                    worker_ratio *= 1 - args.d
                    compressed_part += worker_ratio
                    remain_msg -= 1
            assert remain_msg == args.w
            # print("remain_msg is", remain_msg, "path is", path)
            compressed_part += remain_msg
            avg_compression.append(compressed_part / total_msg)
        assert len(avg_compression) == len(paths)
        std, mean = torch.std_mean(torch.tensor(avg_compression))
        print("compression", mean, std)
        print("co", mean * total_msg, std * total_msg)
