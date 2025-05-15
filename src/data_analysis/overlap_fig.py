import gzip, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import numpy as np

# === Path ===
uploaded_path = "data/processed/modelcard_citation_all_matrices.pkl.gz"

with gzip.open(uploaded_path, "rb") as f:
    data = pickle.load(f)

for k, v in data.items():
    if isinstance(v, csr_matrix):
        print(f"{k}: shape={v.shape}, nnz={v.nnz}")

matrix_keys = [k for k, v in data.items() if isinstance(v, csr_matrix)]

# === Layout configuration ===
n_cols = 5
n_rows = 3
fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(4.8 * n_cols, 2.2 * n_rows * 2))
max_plot_size = 50

for idx, key in enumerate(matrix_keys[:n_cols * n_rows]):
    row_idx = (idx // n_cols) * 2
    col_idx = idx % n_cols

    matrix = data[key]
    dense_matrix = matrix[:max_plot_size, :max_plot_size].astype(np.float32).todense()

    # --- Heatmap ---
    ax_heatmap = axes[row_idx, col_idx]
    sns.heatmap(dense_matrix, ax=ax_heatmap, cmap="Blues", cbar=False,
                square=True, vmin=0, vmax=1)
    ax_heatmap.set_title(key, fontsize=9)
    ax_heatmap.axis("off")

    # --- Histogram with 0s included ---
    ax_hist = axes[row_idx + 1, col_idx]
    full_dense = matrix.toarray().astype(np.float32).flatten()
    ax_hist.hist(full_dense, bins=50, color="blue")
    ax_hist.set_yscale("log")
    ax_hist.set_title(f"Score dist (incl. 0) of {key}", fontsize=8)
    ax_hist.set_xlabel("")
    ax_hist.set_ylabel("log(Freq)", fontsize=8)
    ax_hist.tick_params(labelsize=7)

# Hide unused plots
total_slots = n_cols * n_rows
for idx in range(len(matrix_keys), total_slots):
    for offset in [0, 1]:
        ax = axes[(idx // n_cols) * 2 + offset, idx % n_cols]
        ax.axis("off")

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
norm = plt.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax)

plt.suptitle("Heatmap + Score Histogram (log scale, incl. 0s)", fontsize=14)
plt.tight_layout(rect=[0, 0, 0.91, 0.96])
plt.savefig("1.png")
plt.show()
