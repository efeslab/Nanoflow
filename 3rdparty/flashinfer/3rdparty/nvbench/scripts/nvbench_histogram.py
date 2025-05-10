#!/usr/bin/env python

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys

from nvbench_json import reader

def parse_files():
    help_text = "%(prog)s [nvbench.out.json | dir/] ..."
    parser = argparse.ArgumentParser(prog='nvbench_histogram', usage=help_text)

    args, files_or_dirs = parser.parse_known_args()

    filenames = []
    for file_or_dir in files_or_dirs:
        if os.path.isdir(file_or_dir):
            for f in os.listdir(file_or_dir):
                if os.path.splitext(f)[1] != ".json":
                    continue
                filename = os.path.join(file_or_dir, f)
                if os.path.isfile(filename) and os.path.getsize(filename) > 0:
                    filenames.append(filename)
        else:
            filenames.append(file_or_dir)

    filenames.sort()

    if not filenames:
        parser.print_help()
        exit(0)

    return filenames


def extract_filename(summary):
    summary_data = summary["data"]
    value_data = next(filter(lambda v: v["name"] == "filename", summary_data))
    assert(value_data["type"] == "string")
    return value_data["value"]


def extract_size(summary):
    summary_data = summary["data"]
    value_data = next(filter(lambda v: v["name"] == "size", summary_data))
    assert(value_data["type"] == "int64")
    return int(value_data["value"])


def parse_samples_meta(filename, state):
    summaries = state["summaries"]
    if not summaries:
        return None, None

    summary = next(filter(lambda s: s["tag"] == "nv/json/bin:nv/cold/sample_times",
                          summaries),
                   None)
    if not summary:
        return None, None

    sample_filename = extract_filename(summary)

    # If not absolute, the path is relative to the associated .json file:
    if not os.path.isabs(sample_filename):
        sample_filename = os.path.join(os.path.dirname(filename), sample_filename)

    sample_count = extract_size(summary)
    return sample_count, sample_filename


def parse_samples(filename, state):
    sample_count, samples_filename = parse_samples_meta(filename, state)
    if not sample_count or not samples_filename:
        return []

    with open(samples_filename, "rb") as f:
        samples = np.fromfile(f, "<f4")

    assert (sample_count == len(samples))
    return samples


def to_df(data):
    return pd.DataFrame.from_dict(dict([(k, pd.Series(v)) for k, v in data.items()]))


def parse_json(filename):
    json_root = reader.read_file(filename)

    samples_data = {}

    for bench in json_root["benchmarks"]:
        print("Benchmark: {}".format(bench["name"]))
        for state in bench["states"]:
            print("State: {}".format(state["name"]))

            samples = parse_samples(filename, state)
            if len(samples) == 0:
                continue

            samples_data["{} {}".format(bench["name"], state["name"])] = samples

    return to_df(samples_data)


def main():
    filenames = parse_files()

    dfs = [parse_json(filename) for filename in filenames]
    df = pd.concat(dfs, ignore_index=True)

    sns.displot(df, rug=True, kind="kde", fill=True)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
