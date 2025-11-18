import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from collections import OrderedDict
sys.path.insert(0, '.')
from src.gpt_evaluation.sparse_matrix_loader import SparseMatrixLoader

plt.rcParams.update({'font.size': 13, 'axes.titlesize': 16, 'axes.labelsize': 15, 'xtick.labelsize': 12, 'ytick.labelsize': 12})

def load_csv_list(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def cohens_kappa(labels1, labels2):
    """计算 Cohen's Kappa"""
    confusion = np.zeros((2, 2))
    for l1, l2 in zip(labels1, labels2):
        confusion[int(l1), int(l2)] += 1
    P_o = (confusion[0, 0] + confusion[1, 1]) / np.sum(confusion)
    P_e = ((confusion[0, :].sum() * confusion[:, 0].sum()) + \
           (confusion[1, :].sum() * confusion[:, 1].sum())) / (np.sum(confusion) ** 2)
    return (P_o - P_e) / (1 - P_e) if P_e != 1 else 1.0

def build_edge_set_from_sparse(loader, loader_csv_list, union_csv_list):
    """
    直接从稀疏矩阵的非零元素构建边集合，避免遍历所有pairs
    返回：set of (union_idx_a, union_idx_b) tuples where union_idx_a < union_idx_b
    """
    # 创建从 loader csv 到 union index 的映射
    csv_to_union_idx = {csv: i for i, csv in enumerate(union_csv_list)}
    
    edge_set = set()
    
    # 直接访问稀疏矩阵的内部结构
    matrix = loader.matrix
    
    # 获取所有非零元素的坐标
    rows, cols = matrix.nonzero()
    
    print(f"    Processing {len(rows)} nonzero entries...")
    
    for idx in range(len(rows)):
        i, j = rows[idx], cols[idx]
        
        # 只处理上三角（避免重复）
        if i >= j:
            continue
            
        # 获取对应的 csv 名称
        csv_i = loader_csv_list[i]
        csv_j = loader_csv_list[j]
        
        # 映射到 union index
        if csv_i in csv_to_union_idx and csv_j in csv_to_union_idx:
            union_i = csv_to_union_idx[csv_i]
            union_j = csv_to_union_idx[csv_j]
            
            # 确保顺序
            if union_i < union_j:
                edge_set.add((union_i, union_j))
            else:
                edge_set.add((union_j, union_i))
    
    return edge_set

def main():
    # Step1 sampling 路径
    paper_nnz_path = "data/gt/csv_pair_matrix_direct_label.npz"
    paper_list_path = "data/gt/csv_list_direct_label.pkl"
    mc_nnz_path = "data/gt/scilake_gt_modellink_model_adj_processed.npz"
    mc_list_path = "data/gt/scilake_gt_modellink_model_adj_csv_list_processed.pkl"
    ds_nnz_path = "data/gt/scilake_gt_modellink_dataset_adj_processed.npz"
    ds_list_path = "data/gt/scilake_gt_modellink_dataset_adj_csv_list_processed.pkl"

    print("Loading csv lists...")
    csv_list_paper = load_csv_list(paper_list_path)
    csv_list_mc = load_csv_list(mc_list_path)
    csv_list_ds = load_csv_list(ds_list_path)
    
    # 创建 union csv list (保持顺序)
    print("Creating union csv list...")
    union_csv_set = OrderedDict()
    for csv in csv_list_paper:
        union_csv_set[csv] = None
    for csv in csv_list_mc:
        union_csv_set[csv] = None
    for csv in csv_list_ds:
        union_csv_set[csv] = None
    
    union_csv_list = list(union_csv_set.keys())
    N_union = len(union_csv_list)
    print(f"Union size: {N_union} CSVs")
    print(f"  Paper: {len(csv_list_paper)} CSVs")
    print(f"  ModelCard: {len(csv_list_mc)} CSVs")
    print(f"  Dataset: {len(csv_list_ds)} CSVs")
    
    # 创建各自的映射
    csv_to_idx_paper = {c: i for i, c in enumerate(csv_list_paper)}
    csv_to_idx_mc = {c: i for i, c in enumerate(csv_list_mc)}
    csv_to_idx_ds = {c: i for i, c in enumerate(csv_list_ds)}
    
    # 加载 loaders
    print("Loading sparse matrices...")
    paper_loader = SparseMatrixLoader(paper_nnz_path)
    mc_loader = SparseMatrixLoader(mc_nnz_path)
    ds_loader = SparseMatrixLoader(ds_nnz_path)
    
    # 构建边集合（直接从稀疏矩阵的非零元素）
    print("Building edge sets from sparse matrices...")
    print("  Building paper edge set...")
    paper_edges = build_edge_set_from_sparse(paper_loader, csv_list_paper, union_csv_list)
    print(f"    Found {len(paper_edges)} unique edges")
    
    print("  Building modelcard edge set...")
    mc_edges = build_edge_set_from_sparse(mc_loader, csv_list_mc, union_csv_list)
    print(f"    Found {len(mc_edges)} unique edges")
    
    print("  Building dataset edge set...")
    ds_edges = build_edge_set_from_sparse(ds_loader, csv_list_ds, union_csv_list)
    print(f"    Found {len(ds_edges)} unique edges")
    
    # 只考虑至少在一个矩阵中出现的 pairs（而不是所有可能的pairs）
    print("Computing union of all edge pairs...")
    all_relevant_pairs = paper_edges | mc_edges | ds_edges
    print(f"Total relevant pairs (union): {len(all_relevant_pairs)}")
    
    # 为每个 pair 创建标签
    print("Creating labels for relevant pairs...")
    gt_level_labels = {
        'paper': [],
        'modelcard': [],
        'dataset': []
    }
    
    edge_sets = {
        'paper': paper_edges,
        'modelcard': mc_edges,
        'dataset': ds_edges
    }
    
    for pair in sorted(all_relevant_pairs):
        for level in ['paper', 'modelcard', 'dataset']:
            label = 1 if pair in edge_sets[level] else 0
            gt_level_labels[level].append(label)
    
    print(f"Created {len(gt_level_labels['paper'])} pair labels")
    
    # 计算 Cohen's Kappa matrix
    print("Computing Cohen's Kappa...")
    gt_lvls = ['paper', 'modelcard', 'dataset']
    gt_names = ['Paper', 'ModelCard', 'Dataset']
    gt_matrix = np.zeros((3, 3))
    
    for x in range(3):
        for y in range(3):
            kappa = cohens_kappa(gt_level_labels[gt_lvls[x]], gt_level_labels[gt_lvls[y]])
            gt_matrix[x, y] = kappa
            print(f"  {gt_names[x]} vs {gt_names[y]}: {kappa:.3f}")
    
    # 可视化
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(gt_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=gt_names, yticklabels=gt_names, 
                square=True, cbar=True, annot_kws={'size': 14},
                vmin=0, vmax=1)
    ax.set_title("Sampling GT-GT Consistency (Kappa)", fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig("data/analysis/sampling_gt_gt_agreement.pdf", bbox_inches='tight', dpi=300)
    print("✓ Saved to data/analysis/sampling_gt_gt_agreement.pdf")
    plt.close()
    
    # 保存结果到文件
    results = {
        'kappa_matrix': gt_matrix.tolist(),
        'level_names': gt_names,
        'n_union_csvs': N_union,
        'n_relevant_pairs': len(all_relevant_pairs),
        'n_edges': {
            'paper': len(paper_edges),
            'modelcard': len(mc_edges),
            'dataset': len(ds_edges)
        }
    }
    
    output_path = "output/gpt_evaluation/sampling_gt_consistency.json"
    import json
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {output_path}")

if __name__ == "__main__":
    main()

