import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = "/code/Nanoflow-python/profile_data/Llama3-8B/KQV.db"
TABLE   = "torch"  # <-- change this


# ---- configurable constants for the optimal line ----
M = 2048
N = 4096
K = 6144
FLOPS = 989e12
# ----------------------------------------------------

# Optional: lock other params so lines only reflect sm_count and batch_size
LOCK = {
    # "seq_len": 1024,
    # "head_dim": 128,
    # "num_qo_heads": 32,
    # "num_kv_heads": 8,
}

where = " AND ".join([f"{k} = :{k}" for k in LOCK]) or "1=1"

q = f"""
SELECT
  sm_count,
  batch_size,
  AVG(average_time_ms) AS avg_time_ms
FROM {TABLE}
WHERE {where}
GROUP BY sm_count, batch_size
ORDER BY sm_count, batch_size;
"""

with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql_query(q, conn, params=LOCK)
print("Finish querying database")

# Pivot so each batch_size becomes a line
pivot = df.pivot(index="sm_count", columns="batch_size", values="avg_time_ms").sort_index()



plt.figure(figsize=(7, 4))
for bsz in pivot.columns:
    if bsz % 512 == 0:
        plt.plot(pivot.index, pivot[bsz], marker="o", label=f"batch_size={bsz}")

plt.title("Duration vs SM count for Torch::GEMM")
plt.xlabel("SM count")
plt.ylabel("Duration (ms)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Lines by batch size")
plt.tight_layout()
plt.savefig("llama3-8B_KQV.pdf")

# float16(2) * kv_head_num(16) * head_dim(128) * seq_len(1024) * batch_size / Mem bandwidth(4.8 * 1e12)
optimal_time_ms = (
    2*M*K*N / FLOPS
) * 1000.0

# another figure
plt.figure(figsize=(7, 4))

# Suppose you only want batch_size = 2048
pivot_2048 = pivot[[2048]]
x_sm = pivot_2048.index.to_numpy()

# Real curve
for bsz in pivot_2048.columns:
    plt.plot(pivot_2048.index, pivot_2048[bsz], marker="o", label=f"batch_size={bsz}")
# Optimal line (dotted)
plt.plot(
    df["sm_count"],
    [optimal_time_ms] * len(df),
    linestyle=":",
    label=f"Optimal ≈ {optimal_time_ms:.3f} ms"
)

# Scaling curve (baseline from sm=132)
if 132 in pivot_2048.index:
    t_at_132 = float(pivot_2048.loc[132, 2048])
else:
    raise ValueError("No data point at sm=132 to base the scaling curve on.")

scale_curve = t_at_132 * 132.0 / x_sm
plt.plot(x_sm, scale_curve, linestyle="--", label="Scaled from sm=132")

plt.title("Duration vs SM Count (batch=2048)")
plt.xlabel("SM count")
plt.ylabel("Duration (ms)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.ylim(top=2.0)
plt.tight_layout()
plt.savefig("llama3-8B_KQV_2048.pdf")