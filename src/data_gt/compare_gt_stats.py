import os, pickle, hashlib
from collections import Counter

def load_gt(path):
    """Load a pickled adjacency dict."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def hash_file(path):
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def summarize(adj):
    """Return number of nodes, edges, avg degree."""
    num_nodes = len(adj)
    total_links = sum(len(v) for v in adj.values())
    avg_degree = total_links / num_nodes if num_nodes else 0
    return num_nodes, total_links // 2, avg_degree


def main(gt_dir):
    LEVELS = [
        "direct_label",
        "direct_label_influential",
        "direct_label_methodology_or_result",
        "direct_label_methodology_or_result_influential",
        "max_pr",
        "max_pr_influential",
        "max_pr_methodology_or_result",
        "max_pr_methodology_or_result_influential",
    ]

    adjs = {}
    hashes = {}
    for lvl in LEVELS:
        fname = f"csv_pair_adj_{lvl}_processed.pkl"
        path = os.path.join(gt_dir, fname)
        if not os.path.exists(path):
            print(f"{lvl}: MISSING")
            continue
        print(f"loading {path}")
        adjs[lvl] = load_gt(path)
        hashes[lvl] = hash_file(path)

    # 1) File hashes
    print("\nFile MD5 hashes:")
    for lvl, h in hashes.items():
        print(f"  {lvl}: {h}")

    # 2) Basic stats
    print("\nBasic GT stats:")
    print(f"{'Level':<40}{'Nodes':>8}{'Edges':>10}{'AvgDeg':>10}")
    print('-' * 70)
    base = LEVELS[0]
    for lvl, adj in adjs.items():
        nodes, edges, avg_deg = summarize(adj)
        print(f"{lvl:<40}{nodes:>8}{edges:>10}{avg_deg:>10.2f}")

    # 3) Equality checks against base level
    print(f"\nComparisons to '{base}':")
    for lvl in LEVELS[1:]:
        if lvl in adjs:
            equal = (adjs[lvl] == adjs[base])
            print(f"  {lvl} == {base}: {equal}")
            if not equal:
                extra = set(adjs[lvl].keys()) - set(adjs[base].keys())
                missing = set(adjs[base].keys()) - set(adjs[lvl].keys())
                print(f"    extra keys: {sorted(list(extra))[:5]}...")
                print(f"    missing keys: {sorted(list(missing))[:5]}...")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare GT adjacency pickle files with deeper debug')
    parser.add_argument('--gt_dir', type=str, default='data/gt', help='Directory of GT pickles')
    args = parser.parse_args()
    main(args.gt_dir)

