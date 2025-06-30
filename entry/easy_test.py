import pickle
import sys, os
import torch
import time
sys.path.append("../")
sys.path.append("../pybind/build")
# os.environ["HF_HOME"] = "/code/hf"

# import bind_genEmbedding
import bind_gemm

torch.cuda.set_device(0)

stream = torch.cuda.Stream()
stream_handle = stream.cuda_stream


# O_A = torch.load(os.path.join(path0, "O_0_A")).cuda()
# O_B = torch.load(os.path.join(path0, "O_0_None")).cuda()
# O_C = torch.load(os.path.join(path0, "O_0_C"))
# out = torch.zeros((O_A.shape[0], O_B.shape[1]), dtype=torch.float16, device='cuda')

# offset = 0
# stride = 4096
# print("GEMM_N_Parallel_Layer run", "offset", offset, "stride", stride)
# C = O_C[:, offset: offset + stride].cuda()

# print(O_A.matmul(O_B) + C)

# impl_tag_profile = "SM90_128_256_64_1_2_1_1_RowMajor_RowMajor_RowMajor_auto"
# print("M:", O_A.shape[0], "N:", O_B.shape[1], "K:", O_B.shape[0])
# bind_gemm.configGEMM(impl_tag_profile, "O", O_A.shape[0], O_B.shape[1], O_B.shape[0], 1.0, 1.0)

# bind_gemm.gemmLauncher("O", O_A, O_B, C, out, stream_handle)

# print("out", out)

# tensor_pool = torch.load(os.path.join(path3, "output.pt"))

# GenEmbedding_A = tensor_pool["GenEmbedding_0_token"]
# GenEmbedding_B = tensor_pool["GenEmbedding_0_None"]

# print("GenEmbedding_A", GenEmbedding_A)
# print("GenEmbedding_B", GenEmbedding_B)

print([384]*14)