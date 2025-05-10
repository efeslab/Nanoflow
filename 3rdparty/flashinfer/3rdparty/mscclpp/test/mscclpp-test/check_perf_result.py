# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging


def load_perf_file(perf_fine: str) -> dict:
    res = {}
    with open(perf_fine, "r") as f:
        for line in f:
            data = json.loads(line)
            res[(data["name"], data["kernel"], data["ranks"], data["ranksPerNode"], data["size"])] = {
                "algBw": data["algBw"],
                "busBw": data["busBw"],
                "time": data["time"],
            }
            if "target" in data:
                res[(data["name"], data["kernel"], data["ranks"], data["ranksPerNode"], data["size"])]["target"] = data[
                    "target"
                ]
    return res


def check_perf_result(perf_result: dict, baseline: dict, time_threshold: float, bandwidth_threshold: float) -> bool:
    res = True
    threshold = None
    for key, value in perf_result.items():
        if key not in baseline:
            continue
        if baseline[key]["target"] == "latency":
            threshold = time_threshold
        else:
            threshold = bandwidth_threshold
        if abs(value["time"] - baseline[key]["time"]) / baseline[key]["time"] > threshold:
            logging.error(
                "%s: time %f not match baseline %f with threshold %f",
                str(key),
                value["time"],
                baseline[key]["time"],
                threshold,
            )
            res = False
    return res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--perf-file", type=str, required=True)
    parser.add_argument("--baseline-file", type=str, required=True)
    # We use different threshold for latency and bandwidth. For latency,
    # small data size is used which introduces more variance. For bandwidth, the performance is more stable.
    parser.add_argument("--time-threshold", type=float, default=0.15)
    parser.add_argument("--bandwidth-threshold", type=float, default=0.05)
    args = parser.parse_args()

    perf_result = load_perf_file(args.perf_file)
    baseline = load_perf_file(args.baseline_file)
    if check_perf_result(perf_result, baseline, args.time_threshold, args.bandwidth_threshold):
        print("PASS")
    else:
        print("FAIL")
        exit(1)
