from hip import hip, hipblas
from hip.hip import hipExtStreamCreateWithCUMask
import torch
import logging

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hipblas.hipblasStatus_t)
        and err != hipblas.hipblasStatus_t.HIPBLAS_STATUS_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result

def create_streams_with_cumask(cu_counts: list[int], device_id: str) -> list[torch.Stream]:
    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, int(device_id.split(":")[-1])))
    total_cus = props.multiProcessorCount
    mask_size = (total_cus + 31) // 32

    streams = []
    bit_offset = 0

    for count in cu_counts:
        if bit_offset >= total_cus:
            break
        take = min(count, total_cus - bit_offset)

        mask = [0] * mask_size
        for i in range(take):
            bit = bit_offset + i
            mask[bit // 32] |= (1 << (bit % 32))

        logging.debug([bin(b) for b in mask])
        s = hip_check(hip.hipExtStreamCreateWithCUMask(mask_size, mask))
        streams.append(torch.cuda.get_stream_from_external(s))
        bit_offset += take

    return streams
