import torch
from torch.cuda.streams import ExternalStream
import bind_green_ctx

def create_greenctx(
    sm_a: float, sm_b: float, device_id: int
) -> tuple[ExternalStream, ExternalStream, int, int]:
    """Create two green context streams based on the specified percentages of sm_a and sm_b."""
    res = bind_green_ctx.create_greenctx_stream_by_percent(sm_a, sm_b, device_id)
    stream_a = ExternalStream(
        stream_ptr=res[0], device=torch.device(f"cuda:{device_id}")
    )
    stream_b = ExternalStream(
        stream_ptr=res[1], device=torch.device(f"cuda:{device_id}")
    )
    return stream_a, stream_b, res[2], res[3]