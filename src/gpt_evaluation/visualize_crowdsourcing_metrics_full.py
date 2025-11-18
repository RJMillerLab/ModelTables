import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, '.')
from src.gpt_evaluation.sparse_matrix_loader import SparseMatrixLoader

plt.rcParams.update({'font.size': 13, 'axes.titlesize': 16, 'axes.labelsize': 15, 'xtick.labelsize': 12, 'ytick.labelsize': 12})

def load_csv_list(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def cohens_kappa(labels1, labels2):
    confusion = np.zeros((2, 2))
    for l1, l2 in zip(labels1, labels2):
        confusion[int(l1), int(l2)] += 1
    P_o = (confusion[0, 0] + confusion[1, 1]) / np.sum(confusion)
    P_e = ((confusion[0, :].sum() * confusion[:, 0].sum()) + \
           (confusion[1, :].sum() * confusion[:, 1].sum())) / (np.sum(confusion) ** 2)
    return (P_o - P_e) / (1 - P_e) if P_e != 1 else 1.0

def main():
    # 路径
    paper_nnz_path = "data/gt/csv_pair_matrix_direct_label.npz"
    paper_list_path = "data/gt/csv_list_direct_label.pkl"
    mc_nnz_path = "data/gt/scilake_gt_modellink_model_adj_processed.npz"
    mc_list_path = "data/gt/scilake_gt_modellink_model_adj_csv_list_processed.pkl"
    ds_nnz_path = "data/gt/scilake_gt_modellink_dataset_adj_processed.npz"
    ds_list_path = "data/gt/scilake_gt_modellink_dataset_adj_csv_list_processed.pkl"

    # 以paper为主保证顺序一致
    csv_list_paper = load_csv_list(paper_list_path)
    csv_list_mc = load_csv_list(mc_list_path)
    csv_list_ds = load_csv_list(ds_list_path)
    N = len(csv_list_paper)

    # index->csv 映射
    csv2idx_mc = {c:i for i, c in enumerate(csv_list_mc)}
    csv2idx_ds = {c:i for i, c in enumerate(csv_list_ds)}

    paper_loader = SparseMatrixLoader(paper_nnz_path)
    mc_loader = SparseMatrixLoader(mc_nnz_path)
    ds_loader = SparseMatrixLoader(ds_nnz_path)

    gt_level_labels = {'paper': [], 'modelcard': [], 'dataset': []}
    for i in range(N):
        for j in range(i+1, N):
            # 三个 index 映射对齐
            c_a, c_b = csv_list_paper[i], csv_list_paper[j]
            i_mc = csv2idx_mc.get(c_a, None)
            j_mc = csv2idx_mc.get(c_b, None)
            i_ds = csv2idx_ds.get(c_a, None)
            j_ds = csv2idx_ds.get(c_b, None)
            # 跳过不存在pair的label
            lab_paper = int(paper_loader.has_edge(i, j))
            if i_mc is not None and j_mc is not None:
                lab_mc = int(mc_loader.has_edge(i_mc, j_mc))
            else:
                lab_mc = 0
            if i_ds is not None and j_ds is not None:
                lab_ds = int(ds_loader.has_edge(i_ds, j_ds))
            else:
                lab_ds = 0
            gt_level_labels['paper'].append(lab_paper)
            gt_level_labels['modelcard'].append(lab_mc)
            gt_level_labels['dataset'].append(lab_ds)

    gt_lvls = ['paper', 'modelcard', 'dataset']
    gt_names = ['Paper', 'ModelCard', 'Dataset']
    gt_matrix = np.zeros((3, 3))
    for x in range(3):
        for y in range(3):
            gt_matrix[x, y] = cohens_kappa(gt_level_labels[gt_lvls[x]], gt_level_labels[gt_lvls[y]])

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    sns.heatmap(gt_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=gt_names, yticklabels=gt_names, square=True, cbar=True, annot_kws={'size': 13})
    ax.set_title("Global GT-GT Consistency (Kappa)", fontsize=15)
    plt.tight_layout()
    plt.savefig("data/analysis/global_gt_gt_agreement.pdf", bbox_inches='tight', dpi=300)
    print("✓ Saved global GT-GT to data/analysis/global_gt_gt_agreement.pdf")
    plt.close()

if __name__ == "__main__":
    main()
