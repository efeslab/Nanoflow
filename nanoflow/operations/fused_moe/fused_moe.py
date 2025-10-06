from numpy import isin
import torch
import torch.nn.functional as F
import time
import sqlite3

import nanoflow.platform_config as platform_config
from nanoflow.operations import Operations, Operation_Layer, OperationImpl
from nanoflow.core import IOWrapper, WeightWrapper, process_weight_list
from nanoflow.utils.prof_marker import prof_marker

from flashinfer.fused_moe import cutlass_fused_moe


def compute_routing(
    router_logits: torch.Tensor, top_k: int, norm_topk_prob: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute routing weights and selected experts from router logits.

    Args:
        router_logits (torch.Tensor): Router logits of shape [batch_size, num_experts]
        top_k (int): Number of experts to route to per token

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - routing_weights: Expert weights of shape [batch_size, top_k]
            - selected_experts: Expert indices of shape [batch_size, top_k]
    """
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    return routing_weights, selected_experts


class FusedMoETorchImpl(OperationImpl):
    category_tag = "torch"

    def run(
        self,
        x,
        router_logits,
        w31_weight,
        w2_weight,
        out,
        num_experts,
        top_k,
        norm_topk_prob,
    ):
        with torch.cuda.stream(self.stream):
            routing_weights, selected_experts = compute_routing(router_logits, top_k, norm_topk_prob)
            results = torch.zeros_like(x)
            for expert_id in range(num_experts):
                mask = selected_experts == expert_id
                if not mask.sum():
                    continue
                batch_idx, nth_expert = torch.where(mask)
                w31_expert = w31_weight[
                    expert_id
                ]  # [2 * intermediate_size, hidden_size]
                w2_expert = w2_weight[expert_id]  # [hidden_size, intermediate_size]

                # Split w13 into w1 and w3
                w3_expert, w1_expert = torch.chunk(w31_expert, 2, dim=0)

                expert_inputs = x[batch_idx]
                inter = F.silu(expert_inputs @ w1_expert.t()) * (
                    expert_inputs @ w3_expert.t()
                )
                output = inter @ w2_expert.t()
                results[batch_idx] += (
                    routing_weights[batch_idx, nth_expert, None] * output
                )
            out.copy_(results)


class FusedMoEImpl(OperationImpl):
    category_tag = "cutlass"

    def run(self, x, router_logits, W31, W2, output, num_experts, top_k, norm_topk_prob):
        with torch.cuda.stream(self.stream):
            routing_weights, selected_experts = compute_routing(router_logits, top_k, norm_topk_prob)
            flash_output = cutlass_fused_moe(
                x,
                selected_experts.to(torch.int),
                routing_weights,
                W31,
                W2,
                output.dtype,
                output=output,
                quant_scales=None,
            )


class FusedMoE(Operations):
    def __init__(self, name, device, nano_idx=None):
        super().__init__(name, device, nano_idx)
        self.inputs = {
            "x": IOWrapper(self, "x", device).is_input(),  # [batch, hidden_dim]
            "router_logits": IOWrapper(
                self, "router_logits", device
            ).is_input(),  # [batch, num_experts]
        }
        self.outputs = {"output": IOWrapper(self, "output", device).is_output()}
        self.weights = {
            "W31": WeightWrapper(self, name="W31"),
            "W2": WeightWrapper(self, name="W2"),
        }

        self.init_impl_map()
        self.op_layer = FusedMoE_Layer

    def init_impl_map(self):
        self.impl_map = {}
        self.add_impl(FusedMoETorchImpl)
        self.add_impl(FusedMoEImpl)

    def setShape(
        self, num_experts, moe_intermediate_dim, hidden_dim, top_k, norm_topk_prob
    ):
        self.num_experts = num_experts
        self.moe_intermediate_dim = moe_intermediate_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.inputs["x"].init_shape((0, hidden_dim))
        self.inputs["router_logits"].init_shape((0, num_experts))
        self.outputs["output"].init_shape((0, hidden_dim))
        self.weights["W31"].shape = (num_experts, 2 * moe_intermediate_dim, hidden_dim)
        self.weights["W2"].shape = (num_experts, hidden_dim, moe_intermediate_dim)

        return self

    def run(self, layer):
        self.impl.run(
            self.inputs["x"].tensor,
            self.inputs["router_logits"].tensor,
            self.weights["W31"].weight_map[layer],
            self.weights["W2"].weight_map[layer],
            self.outputs["output"].tensor,
            self.num_experts,
            top_k=self.top_k,
            norm_topk_prob=self.norm_topk_prob,
        )

    def processWeight(self, global_weight_map, cached_weight_map, cached, device):
        if not cached:
            weight_name = self.weight_name
            num_weights = len(self.weights)
            if not isinstance(weight_name, list):
                weight_name = [weight_name]
            assert num_weights == 2, f"num_weights = {num_weights}, should be 2"
            assert (
                len(weight_name) == 2
            ), f"len(weight_name) = {len(weight_name)}, should be 2"

            # Process W31
            weight_wrapper = self.weights["W31"]
            ug_weights = weight_name[0]
            for l in self.layer_list:
                weights_list = []
                for expert_id in range(self.num_experts):
                    ug_weights_expert = ug_weights[expert_id]
                    weights_combine_ug = []
                    for name in ug_weights_expert:
                        weights_combine_ug.append(
                            global_weight_map[name.format(layer=l)]
                        )
                    weights_list.append(
                        torch.cat(weights_combine_ug, dim=0).contiguous()
                    )
                final_ug = torch.stack(weights_list, dim=0)
                assert (
                    final_ug.shape == weight_wrapper.shape
                ), f"layer = {l}, expected shape = {weight_wrapper.shape}, real shape = {final_ug.shape}"
                cached_weight_map[
                    f"{weight_wrapper.owner.name}_{weight_wrapper.name}_layer_{l}"
                ] = final_ug

            # Process W2
            weight_wrapper = self.weights["W2"]
            down_weights = weight_name[1]
            for l in self.layer_list:
                weights_list = []
                for expert_id in range(self.num_experts):
                    weights_list.append(
                        global_weight_map[down_weights[expert_id].format(layer=l)]
                    )
                final_down = torch.stack(weights_list, dim=0)
                assert (
                    final_down.shape == weight_wrapper.shape
                ), f"layer = {l}, expected shape = {weight_wrapper.shape}, real shape = {final_down.shape}"
                cached_weight_map[
                    f"{weight_wrapper.owner.name}_{weight_wrapper.name}_layer_{l}"
                ] = final_down
        elif cached:
            for weight_wrapper in self.weights.values():
                for l in self.layer_list:
                    weight_wrapper.weight_map[l] = cached_weight_map[
                        f"{weight_wrapper.owner.name}_{weight_wrapper.name}_layer_{l}"
                    ].to(device, non_blocking=True)
                    assert (
                        weight_wrapper.weight_map[l].shape == weight_wrapper.shape
                    ), f"name = {weight_wrapper.name}, expected shape = {weight_wrapper.shape}, layer = {l}, real shape = {weight_wrapper.weight_map[l].shape}"


class FusedMoE_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)

    def run(self):
        self.parent.run(self.layer)
