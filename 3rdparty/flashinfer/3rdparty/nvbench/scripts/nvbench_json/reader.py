import json

from . import version


def read_file(filename):
    with open(filename, "r") as f:
        file_root = json.load(f)
    version.check_file_version(filename, file_root)
    return file_root
