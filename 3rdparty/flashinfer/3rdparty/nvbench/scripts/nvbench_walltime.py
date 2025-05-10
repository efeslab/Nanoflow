#!/usr/bin/env python

import argparse
import math
import os
import sys

from nvbench_json import reader

import tabulate


# Parse version string into tuple, "x.y.z" -> (x, y, z)
def version_tuple(v):
    return tuple(map(int, (v.split("."))))


tabulate_version = version_tuple(tabulate.__version__)

all_devices = []


def format_axis_value(axis_value, axis_type):
    if axis_type == "int64":
        return "%d" % int(axis_value)
    elif axis_type == "float64":
        return "%.5g" % float(axis_value)
    else:
        return axis_value


def format_walltime(seconds_in):
    h = math.floor(seconds_in / (60 * 60))
    m = math.floor((seconds_in / 60) % 60)
    s = math.floor(seconds_in % 60)
    ms = math.floor((seconds_in * 1000) % 1000)

    return "{}{}{}{}".format(
        "{:0>2d}:".format(h) if h > 1e-9 else "",
        "{:0>2d}:".format(m) if (h > 1e-9 or m > 1e-9) else "",
        "{:0>2d}.".format(s) if (h > 1e-9 or m > 1e-9) else "{:d}.".format(s),
        "{:0>3d}".format(ms))


def format_percentage(percentage):
    # When there aren't enough samples for a meaningful noise measurement,
    # the noise is recorded as infinity. Unfortunately, JSON spec doesn't
    # allow for inf, so these get turned into null.
    if percentage is None:
        return "inf"
    return "%0.2f%%" % (percentage * 100.0)


measure_names = ["cold", "batch", "cupti"]
measure_column_names = {"cold": "Isolated", "batch": "Batch", "cupti": "CUPTI"}


def init_measures():
    out = {}
    for name in measure_names:
        out[name] = 0.
    return out


def get_measures(state):
    summaries = state["summaries"]
    times = {}
    for name in measure_names:
        measure_walltime_tag = "nv/{}/walltime".format(name)
        summary = next(filter(lambda s: s["tag"] == measure_walltime_tag,
                              summaries),
                       None)
        if not summary:
            continue

        walltime_data = next(filter(lambda d: d["name"] == "value", summary["data"]))
        assert(walltime_data["type"] == "float64")
        walltime = walltime_data["value"]
        walltime = float(walltime)
        times[name] = walltime if walltime else 0.
    return times


def merge_measures(target, src):
    for name, src_val in src.items():
        target[name] += src_val


def sum_measures(measures):
    total_time = 0.
    for time in measures.values():
        total_time += time
    return total_time


def get_active_measure_names(measures):
    names = []
    for name, time in measures.items():
        if time > 1e-9:
            names.append(name)
    return names


def append_measure_headers(headers, active=measure_names):
    for name in active:
        headers.append(measure_column_names[name])


def append_measure_values(row, measures, active=measure_names):
    for name in active:
        row.append(format_walltime(measures[name]))


def consume_file(filename):
    file_root = reader.read_file(filename)

    file_out = {}
    file_measures = init_measures()

    benches = {}
    for bench in file_root["benchmarks"]:
        bench_data = consume_benchmark(bench, file_root)
        merge_measures(file_measures, bench_data["measures"])
        benches[bench["name"]] = bench_data

    file_out["benches"] = benches
    file_out["measures"] = file_measures
    return file_out


def consume_benchmark(bench, file_root):
    bench_out = {}

    # Initialize axis map
    axes_out = {}
    axes = bench["axes"]
    if axes:
        for axis in axes:
            values_out = {}
            axis_name = axis["name"]
            axis_type = axis["type"]
            for value in axis["values"]:
                if axis_type == "type":
                    value = value["input_string"]
                else:
                    value = format_axis_value(value["value"], axis_type)
                values_out[value] = {"measures": init_measures()}
            axes_out[axis_name] = values_out

    states_out = {}
    bench_measures = init_measures()

    for state in bench["states"]:
        state_name = state["name"]
        # Get walltimes for each measurement:
        state_measures = get_measures(state)
        state_out = {}
        state_out["measures"] = state_measures
        states_out[state_name] = state_out

        # Update the benchmark measures walltimes
        merge_measures(bench_measures, state_measures)

        # Update the axis measurements:
        axis_values = state["axis_values"]
        if axis_values:
            for axis_value in axis_values:
                axis_name = axis_value["name"]
                value = format_axis_value(axis_value["value"], axis_value["type"])
                merge_measures(axes_out[axis_name][value]["measures"], state_measures)

    bench_out["axes"] = axes_out
    bench_out["measures"] = bench_measures
    bench_out["states"] = states_out
    return bench_out


def print_overview_section(data):
    print("# Walltime Overview\n")

    measures = data["measures"]
    active_measures = get_active_measure_names(measures)

    headers = ["Walltime"]
    append_measure_headers(headers, active_measures)

    colalign = ["right"] * len(headers)

    rows = []

    row = [format_walltime(sum_measures(measures))]
    append_measure_values(row, measures, active_measures)
    rows.append(row)

    # colalign and github format require tabulate 0.8.3
    if tabulate_version >= (0, 8, 3):
        print(tabulate.tabulate(rows,
                                headers=headers,
                                colalign=colalign,
                                tablefmt="github"))
    else:
        print(tabulate.tabulate(rows,
                                headers=headers,
                                tablefmt="markdown"))

    print()


# append_data_row_lambda args: (row_list, name, items[name])
def print_measures_table(headers, colalign, items, total_measures, append_item_row_lambda):
    total_time = sum_measures(total_measures)
    active_measures = get_active_measure_names(total_measures)
    num_user_columns = len(headers)

    headers.append("%")
    headers.append("Walltime")
    append_measure_headers(headers, active_measures)

    while len(colalign) < len(headers):
        colalign.append("right")

    rows = []

    for name, item in items.items():
        item_measures = item["measures"]
        item_time = sum_measures(item_measures)

        row = []
        append_item_row_lambda(row, name, item)
        if total_time > 1e-9:
            row.append(format_percentage(item_time / total_time))
        else:
            row.append(format_percentage(0))
        row.append(format_walltime(item_time))
        append_measure_values(row, item_measures, active_measures)
        rows.append(row)

    # Totals:
    row = []
    if num_user_columns != 0:
        row.append("Total")
    while len(row) < num_user_columns:
        row.append("")
    row.append(format_percentage(1))
    row.append(format_walltime(total_time))
    append_measure_values(row, total_measures, active_measures)
    rows.append(row)

    # colalign and github format require tabulate 0.8.3
    if tabulate_version >= (0, 8, 3):
        print(tabulate.tabulate(rows,
                                headers=headers,
                                colalign=colalign,
                                tablefmt="github"))
    else:
        print(tabulate.tabulate(rows,
                                headers=headers,
                                tablefmt="markdown"))


def print_files_section(data):
    print("# Files\n")

    items = data["files"]
    total_measures = data["measures"]
    headers = ["Filename"]
    colalign = ["left"]

    def append_row(row, name, item):
        row.append(name)

    print_measures_table(headers, colalign, items, total_measures, append_row)
    print()

    for filename, file in items.items():
        print_file_section(filename, file)


def print_file_section(filename, file):
    print("## File: {}\n".format(filename))

    items = file["benches"]
    total_measures = file["measures"]
    headers = ["Benchmark"]
    colalign = ["left"]

    def append_row_name(row, name, item):
        row.append(name)

    print_measures_table(headers, colalign, items, total_measures, append_row_name)
    print()

    for bench_name, bench in items.items():
        print_bench_section(bench_name, bench)


def print_bench_section(bench_name, bench):
    print("### Benchmark: {}\n".format(bench_name))

    # TODO split this up so each axis is a column
    items = bench["states"]
    total_measures = bench["measures"]
    headers = ["Configuration"]
    colalign = ["left"]

    def append_row_name(row, name, item):
        row.append(name)

    print_measures_table(headers, colalign, items, total_measures, append_row_name)
    print()

    for axis_name, axis in bench["axes"].items():
        total_measures = bench["measures"]
        headers = ["Axis: " + axis_name]
        colalign = ["left"]
        print_measures_table(headers, colalign, axis, total_measures, append_row_name)
        print()


def main():
    help_text = "%(prog)s [nvbench.out.json | dir/]..."
    parser = argparse.ArgumentParser(prog='nvbench_walltime', usage=help_text)

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

    data = {}

    files_out = {}
    measures = init_measures()
    for filename in filenames:
        file_data = consume_file(filename)
        merge_measures(measures, file_data["measures"])
        files_out[filename] = file_data

    data["files"] = files_out
    data["measures"] = measures

    print_overview_section(data)
    print_files_section(data)


if __name__ == '__main__':
    sys.exit(main())
