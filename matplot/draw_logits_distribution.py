import torch
import matplotlib.pyplot as plt

def select_top_k_tokens(t: torch.Tensor, k: int):
    # t: 1D tensor of probabilities
    topk = torch.topk(t, k=k, dim=-1)
    return topk.indices, topk.values

def compare_and_plot_bar(indices: torch.Tensor,
                         t1: torch.Tensor,
                         t2: torch.Tensor,
                         seq_idx: int,
                         title: str = "Distribution Comparison (Top-k, Bar)"):
    """
    indices: token ids of selected positions (top-k)
    t1, t2: probabilities for those indices from distribution 1 and 2
            (already aligned; same shape as indices)
    """
    result = {}

    # Move to CPU & flatten
    indices = indices.detach().cpu().flatten()
    t1 = t1.detach().cpu().flatten()
    t2 = t2.detach().cpu().flatten()

    # 1. Shape check
    if t1.shape != t2.shape:
        result["same_shape"] = False
        result["shape_1"] = tuple(t1.shape)
        result["shape_2"] = tuple(t2.shape)
        result["max_abs_diff"] = None
        print("Tensors have different shapes:", result["shape_1"], "vs", result["shape_2"])
        return result

    result["same_shape"] = True
    result["shape"] = tuple(t1.shape)

    # 2. Max absolute difference
    diff = torch.abs(t1 - t2)
    max_abs_diff = diff.max().item()
    result["max_abs_diff"] = max_abs_diff

    print(f"Same shape: {result['shape']}")
    print(f"Max |p1 - p2|: {max_abs_diff:.8f}")

    # 3. Bar plot over rank positions, with token IDs as labels
    k = t1.shape[0]
    x = torch.arange(k)
    width = 0.35

    plt.figure(figsize=(8, 5))

    # dist1 and dist2 side-by-side bars
    plt.bar(x - width/2, t1, width, label="huggingface")
    plt.bar(x + width/2, t2, width, label="ours")

    # Optionally: overlay abs diff as points
    # plt.plot(x, diff, marker="o", linestyle="none", label="|difference|")
    plt.margins(x=0.02)
    plt.plot(x, diff, marker="o", linewidth=1.5, label="|difference|", color="red")
    plt.yscale("log", base=10)
    plt.xticks(x, indices.tolist(), rotation=45)
    plt.xlabel("Token ID (top-k)")
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("logits_distribution_comparison_bar_seq_idx_" + str(seq_idx) + ".pdf")

    return result

# Example usage:
if __name__ == "__main__":
    # Full distributions
    # t1 = torch.rand(4, 4).flatten()
    # t1 = t1 / t1.sum()

    # t2 = t1 + 0.00001 * torch.randn_like(t1)
    # t2 = torch.clamp(t2, min=0)
    # t2 = t2 / t2.sum()

    # # Take top-k from t1, and read corresponding probs from t2 at same indices
    # k = 5
    # top_indices_1, top_values_1 = select_top_k_tokens(t1, k)
    # top_values_2 = t2[top_indices_1]

    # print("top_indices:", top_indices_1)
    # print("top_values_1:", top_values_1)
    # print("top_values_2:", top_values_2)

    # stats = compare_and_plot_bar(top_indices_1, top_values_1, top_values_2,
    #                              title="My Two Distributions (Top-k Bar)")
    # print(stats)

    # sequence_id=50
    t1_seq_idx_50 = torch.load("get_logits_output_cuda:0.pt")
    t1_seq_idx_50 = t1_seq_idx_50[-1]
    t1_seq_idx_50 = torch.softmax(t1_seq_idx_50, dim=-1)
    print("shape of t1:", t1_seq_idx_50.shape)
    t2_seq_idx_50 = torch.load("gt_llama_last_probs.pt")
    print("shape of t2:", t2_seq_idx_50.shape)
    assert t1_seq_idx_50.shape == t2_seq_idx_50.shape
    k = 10
    seq_idx = 50
    top_indices_1, top_values_1 = select_top_k_tokens(t1_seq_idx_50, k)
    top_indices_2, top_values_2 = select_top_k_tokens(t2_seq_idx_50, k)
    print("top_indices_1:", top_indices_1)
    print("top_values_1:", top_values_1)
    print("top_indices_2:", top_indices_2)
    print("top_values_2:", top_values_2)
    assert torch.equal(top_indices_1, top_indices_2)
    stats = compare_and_plot_bar(top_indices_1, top_values_2, top_values_1, seq_idx,
                                 title="Logits Distribution Comparison (Top-k Bar), k=" + str(k) + ", sequence_id=" + str(seq_idx))
    print(stats)

    # sequence_id=20
    t1_seq_idx_20 = torch.load("get_logits_output_cuda:0_seq_idx_20.pt")
    t1_seq_idx_20 = t1_seq_idx_20[-1]
    t1_seq_idx_20 = torch.softmax(t1_seq_idx_20, dim=-1)
    print("shape of t1:", t1_seq_idx_20.shape)
    t2_seq_idx_20 = torch.load("gt_llama_last_probs_seq_idx_20.pt")
    print("shape of t2:", t2_seq_idx_20.shape)

    assert t1_seq_idx_20.shape == t2_seq_idx_20.shape
    k = 10
    seq_idx = 20
    top_indices_1, top_values_1 = select_top_k_tokens(t1_seq_idx_20, k)
    top_indices_2, top_values_2 = select_top_k_tokens(t2_seq_idx_20, k)
    print("top_indices_1:", top_indices_1)
    print("top_values_1:", top_values_1)
    print("top_indices_2:", top_indices_2)
    print("top_values_2:", top_values_2)
    assert torch.equal(top_indices_1, top_indices_2)

    stats = compare_and_plot_bar(top_indices_1, top_values_2, top_values_1, seq_idx,
                                 title="Logits Distribution Comparison (Top-k Bar), k=" + str(k) + ", sequence_id=" + str(seq_idx))
    print(stats)