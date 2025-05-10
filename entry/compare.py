import os
import torch

# Define the two folders
folder1 = "./tp_test_torch_0_folder"
folder2 = "./tp_test_flashinfer_0_folder"

def compare_two_files(file1, file2):
    path1 = os.path.join(folder1, file1)
    path2 = os.path.join(folder2, file2)
    if "_0_" not in file1:
        return
    print(f"Comparing {file1}...")
    try:
        tensor1 = torch.load(path1)
        tensor2 = torch.load(path2)
        print("tensor1 shape:", tensor1.shape)
        print("tensor2 shape:", tensor2.shape)
        # Check if tensors are approximately equal
        if torch.equal(tensor1, tensor2):
            print(f"[MATCH] {file1} is identical")
        elif torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
            print(f"[MATCH] {file1} is approximately identical")
            print("Max difference:", torch.abs(tensor1 - tensor2).max())
            print("Average difference:", torch.abs(tensor1 - tensor2).mean())
            # print("Tensor of gt_out:", tensor1)
            # print("Tensor of out   :", tensor2)
        else:
            print(f"[DIFFER] {file1} is different")
            print("Max difference:", torch.abs(tensor1 - tensor2).max())
            print("Average difference:", torch.abs(tensor1 - tensor2).mean())
            # print("Tensor of gt_out:", tensor1)
            # print("Tensor of out   :", tensor2)

    except Exception as e:
        print(f"Error comparing {file1}: {e}")

def compare_files(file):
    path1 = os.path.join(folder1, file)
    path2 = os.path.join(folder2, file)
    if "_0_" not in file:
        return
    # print(f"Comparing {file}...")
    try:
        tensor1 = torch.load(path1)
        tensor2 = torch.load(path2)

        # compare the tensor line by line
        for i in range(1):
            for j in range(tensor1.shape[1] // 128):
                print(f"Comparing {file} line {i} part {j}...")
                start = j * 128
                end = (j + 1) * 128
                if torch.equal(tensor1[i][start:end], tensor2[i][start:end]):
                    print(f"[MATCH] {file} part {j} is identical")
                elif torch.allclose(tensor1[i][start:end], tensor2[i][start:end], atol=atol, rtol=rtol):
                    print(f"[MATCH] {file} part {j} is approximately identical")
                else:
                    print(f"[DIFFER] {file} part {j} is different")
            if torch.equal(tensor1[i], tensor2[i]):
                print(f"[MATCH] {file} line {i} is identical")

            elif torch.allclose(tensor1[i], tensor2[i], atol=atol, rtol=rtol):
                print(f"[MATCH] {file} line {i} is approximately identical")
            else:
                print(f"[DIFFER] {file} line {i} is different")

        # Check if tensors are approximately equal
        if torch.equal(tensor1, tensor2):
            print(f"[MATCH] {file} is identical")
        elif torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
            print(f"[MATCH] {file} is approximately identical")
            # print("Max difference:", torch.abs(tensor1 - tensor2).max())
            # print("Average difference:", torch.abs(tensor1 - tensor2).mean())
            # print("Tensor of gt_out:", tensor1)
            # print("Tensor of out   :", tensor2)
        else:
            print(f"[DIFFER] {file} is different")
            # print("Max difference:", torch.abs(tensor1 - tensor2).max())
            # print("Average difference:", torch.abs(tensor1 - tensor2).mean())
            # print("Tensor of gt_out:", tensor1)
            # print("Tensor of out   :", tensor2)

    except Exception as e:
        print(f"Error comparing {file}: {e}")


# Get the list of files in both folders
files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))

# Find common files in both folders
common_files = files1.intersection(files2)

# Tolerance values for floating-point comparison
atol = 1e-2  # Absolute tolerance
rtol = 1e-2  # Relative tolerance

# compare_two_files("AllGatherAttn_0_output", "PFAttn_0_output")
compare_files("PFAttn_0_output")
# # Compare each file
# for file in common_files:
#     compare_files(file)

# path1 = os.path.join(folder1, "KQV_0_A")
# path2 = os.path.join(folder1, "KQV_0_None")
# tensor1 = torch.load(path1)
# tensor2 = torch.load(path2)
# result = tensor1.matmul(tensor2)
# print(result)