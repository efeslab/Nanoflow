import os
import torch

# Define the two folders
folder1 = "./gt_out"
folder2 = "./out"

# Get the list of files in both folders
files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))

# Find common files in both folders
common_files = files1.intersection(files2)

# Tolerance values for floating-point comparison
atol = 1e-2  # Absolute tolerance
rtol = 1e-2  # Relative tolerance

# Compare each file
for file in common_files:
    path1 = os.path.join(folder1, file)
    path2 = os.path.join(folder2, file)

    try:
        tensor1 = torch.load(path1)
        tensor2 = torch.load(path2)

        # Check if tensors are approximately equal
        if torch.equal(tensor1, tensor2):
            print(f"[MATCH] {file} is identical")
        elif torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
            print(f"[MATCH] {file} is approximately identical")
            print("Max difference:", torch.abs(tensor1 - tensor2).max())
            print("Average difference:", torch.abs(tensor1 - tensor2).mean())
            print("Tensor of gt_out:", tensor1)
            print("Tensor of out   :", tensor2)
        else:
            print(f"[DIFFER] {file} is different")
            print("Max difference:", torch.abs(tensor1 - tensor2).max())
            print("Average difference:", torch.abs(tensor1 - tensor2).mean())
            print("Tensor of gt_out:", tensor1)
            print("Tensor of out   :", tensor2)


    except Exception as e:
        print(f"Error comparing {file}: {e}")
