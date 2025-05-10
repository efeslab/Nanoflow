# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

import functools
import os


@functools.lru_cache(None)
def library_path():
    return os.path.join(os.path.dirname(__file__), "library")
