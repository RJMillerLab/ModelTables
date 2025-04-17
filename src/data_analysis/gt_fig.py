import gzip, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import numpy as np

print("🔄 Loading GT pickle ...")
GT_PATH = "data/gt/scilake_gt_all_matrices__overlap_rate.pkl.gz"

with gzip.open(GT_PATH, "rb") as f:
    gt_data = pickle.load(f)
print("✅ GT loaded!")

matrix_keys = [
    ("paper_adj", gt_data["paper_adj"]),
    ("model_adj", gt_data["model_adj"]),
    ("csv_adj", gt_data["csv_adj"]),
    ("csv_symlink_adj", gt_data["csv_symlink_adj"]),
    ("csv_real_adj", gt_data["csv_real_adj"]),
]

n_cols = 3
n_rows = (len(matrix_keys) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(5 * n_cols, 2.2 * n_rows * 2))

max_plot_size = 50
dense_limit = 10000  # max shape to convert to dense (to avoid OOM)

for idx, (name, matrix) in enumerate(matrix_keys):
    print(f"\n📊 Processing: {name}")
    row_idx = (idx // n_cols) * 2
    col_idx = idx % n_cols

    # --- Heatmap ---
    ax_heatmap = axes[row_idx, col_idx]
    try:
        if matrix.shape[0] <= dense_limit and matrix.shape[1] <= dense_limit:
            dense_matrix = matrix[:max_plot_size, :max_plot_size].astype(np.float32).todense()
        else:
            dense_matrix = matrix[:max_plot_size, :max_plot_size].astype(np.float32).toarray()
        sns.heatmap(dense_matrix, ax=ax_heatmap, cmap="Blues", cbar=False, square=True, vmin=0, vmax=1)
    except Exception as e:
        print(f"❌ Heatmap failed for {name}: {e}")
        ax_heatmap.text(0.5, 0.5, 'Heatmap Error', ha='center', va='center')
    ax_heatmap.set_title(name, fontsize=9)
    ax_heatmap.axis("off")

    # --- Histogram ---
    ax_hist = axes[row_idx + 1, col_idx]
    try:
        if matrix.shape[0] * matrix.shape[1] <= dense_limit ** 2:
            full_dense = matrix.toarray().astype(np.float32).flatten()
        else:
            full_dense = matrix.data.astype(np.float32)  # sparse only
        ax_hist.hist(full_dense, bins=np.linspace(0, 1.01, 102), color="blue")
        ax_hist.set_yscale("log")
    except Exception as e:
        print(f"❌ Histogram failed for {name}: {e}")
        ax_hist.text(0.5, 0.5, 'Histogram Error', ha='center', va='center')
    ax_hist.set_title(f"Dist of {name}", fontsize=8)
    ax_hist.set_xlabel("")
    ax_hist.set_ylabel("log(Freq)", fontsize=8)
    ax_hist.tick_params(labelsize=7)

# Fill empty plots
total_slots = n_cols * n_rows
for idx in range(len(matrix_keys), total_slots):
    for offset in [0, 1]:
        ax = axes[(idx // n_cols) * 2 + offset, idx % n_cols]
        ax.axis("off")

# Colorbar
print("\n🎨 Adding colorbar ...")
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
norm = plt.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax)

plt.suptitle("GT Matrices: Heatmaps + Histograms (log scale)", fontsize=14)
plt.tight_layout(rect=[0, 0, 0.91, 0.96])

out_path = "gt_matrix_heatmap_histogram.png"
plt.savefig(out_path)
print(f"✅ Saved to {out_path}")
