# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import argparse
from enum import IntEnum
from pathlib import Path
import pkgutil
import sys
from typing import List, Optional

import codegen.ops
from codegen.cmake_config import *


class HandlerId(IntEnum):
    LIST_BLOBS = 0
    WRITE_BLOBS = 1

# inspect all modules under 'codegen.ops' and register API handlers 
ops = []
for importer, module_name, _ in pkgutil.iter_modules(codegen.ops.__path__):
    full_module_name = '%s.%s' % (codegen.ops.__name__, module_name)
    if full_module_name not in sys.modules:
        ops.append(importer.find_spec(module_name).loader.load_module(module_name))
unwanted_prefix = 'fmha_'
handlers = dict(
    [(op.__name__[len(unwanted_prefix):] if op.__name__.startswith(unwanted_prefix) else op.__name__,
        (op.list_blobs, op.write_blobs)) for op in ops]
)
assert 0 < len(handlers)

def write_blobs(output_dir: Optional[str], api_list : List[str], kernel_filter : Optional[str], receipt, mask_impl) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir) / GEN_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    for api in api_list:
        handler = handlers[api][HandlerId.WRITE_BLOBS]
        handler(output_dir, kernel_filter, receipt, mask_impl)

# list all the files that will be generated
def list_blobs(output_file : Optional[str], api_list : List[str], kernel_filter : Optional[str], receipt, mask_impl) -> None:
    assert output_file is not None
    file_path = Path(output_file)

    # create an empty file / drop its contents if it exists
    open(file_path, "w").close()

    for api in api_list:
        handler = handlers[api][HandlerId.LIST_BLOBS]
        handler(file_path, kernel_filter, receipt, mask_impl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK fmha kernel",
    )
    parser.add_argument(
        "-d",
        "--direction", # we keep 'direction' option for backward compatibility
        "-a",
        "--api",
        default='fwd',
        required=False,
        help="supply API(s) to generate (default: fwd). separated by comma."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="write all the blobs into a directory"
    )
    parser.add_argument(
        "-l",
        "--list_blobs",
        required=False,
        help="list all the kernels to a file"
    )
    # TODO: if using filter, must apply same value to output_dir and list_blobs
    parser.add_argument(
        "-f",
        "--filter",
        required=False,
        help="filter out kernels that need to generate, using fnmatch module"
    )

    parser.add_argument(
        "-m",
        "--mask",
        default="simplified",
        required=False,
        help="mask implementation, simplified/generic"
    )

    parser.add_argument(
        "-r",
        "--receipt",
        default=0,
        required=False,
        help="codegen receipt. 0: generate only 8xhdim coverage\n"  + \
             "  1: generate more instance to cover all hdim\n"  + \
             "  2: Only generate instance for Flash attention integration"
    )

    args = parser.parse_args()
    api_list = args.direction.split(',')
    if args.list_blobs is not None:
        list_blobs(args.list_blobs, api_list, args.filter, int(args.receipt), mask_impl=args.mask)
    else:
        write_blobs(args.output_dir, api_list, args.filter, int(args.receipt), mask_impl=args.mask)
