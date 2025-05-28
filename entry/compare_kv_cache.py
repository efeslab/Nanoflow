#!/usr/bin/env python3
import os
import sys
import torch

def find_first_difference(folder1, folder2, prefix="kvcache_0_0_", count=20):
    """
    Compare files named prefix+idx in folder1 and folder2 for idx in [0, count).
    Returns the first idx where files differ, or None if all are identical.
    """
    for idx in range(1, count):
        name = f"{prefix}{idx}.pt"
        path1 = os.path.join(folder1, name)
        path2 = os.path.join(folder2, name)

        # sanity check: files must exist
        if not os.path.isfile(path1):
            print(f"[Error] {path1!r} does not exist.")
            return None
        if not os.path.isfile(path2):
            print(f"[Error] {path2!r} does not exist.")
            return None

        # read and compare
        data1 = torch.load(path1, map_location='cpu')
        data2 = torch.load(path2, map_location='cpu')
        print("max diff:", torch.max(torch.abs(data1 - data2)))
        if not torch.allclose(data1, data2, rtol=1e-01, atol=1e-01):
            print(f"First difference at idx={idx}:")
            print(f"  {path1}")
            print(data1)
            print(f"  {path2}")
            print(data2)
            
            return idx

    print("All files are identical.")
    return None


folder1 = "./kv_cache_testing"
folder2 = "./kv_cache_debug"

find_first_difference(folder1, folder2, prefix="kvcache_0_0_", count=20)

