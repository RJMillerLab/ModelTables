"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-03
Description: Compute the overlap rate of citation and citing papers based on paperId, and save:
- pairwise overlap scores
- thresholded related paper pairs
- direct citation links (if paperId appears in references of another)
"""

import os
import json
import pandas as pd
import pickle
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations

# === Configuration ===
INPUT_PARQUET = "data/processed/modelcard_citation_enriched.parquet"
OUTPUT_PICKLE_SCORE = "data/processed/modelcard_citation_overlap_by_paperId_score.pickle"
#OUTPUT_PICKLE_THRESHOLD = "data/processed/modelcard_citation_overlap_by_paperId_related.pickle"
OUTPUT_PICKLE_DIRECT = "data/processed/modelcard_citation_direct_relation.pickle"
THRESHOLD = 0.6
MODE = "reference"  # or "citation"

def load_paperId_lists(df, mode):
    paperId_to_ids = {}
    for _, row in df.iterrows():
        pid = row["paperId"]
        assert not pd.isna(pid)
        parsed = json.loads(row["parsed_response"])
        paper_list = parsed["cited_papers"] if mode == "reference" else parsed["citing_papers"]
        if len(parsed["cited_papers"])==0:
            print(f"Warning: {len(parsed['cited_papers'])} cited papers found for {pid}.")
        id_list = [p["paperId"] for p in paper_list]
        paperId_to_ids[pid] = set(id_list)
    return paperId_to_ids

def compute_overlap_scores(paperId_to_ids):
    score_map = {}
    #related_map = defaultdict(set)
    paper_items = list(paperId_to_ids.items())

    for (pid1, set1), (pid2, set2) in tqdm(combinations(paper_items, 2), total=len(paper_items)*(len(paper_items)-1)//2):
        inter = len(set1 & set2)
        union = len(set1) + len(set2)
        if union == 0:
            score = 0
        else:
            score = (2 * inter) / union
        score_map[(pid1, pid2)] = score
        #if score >= THRESHOLD:
            #related_map[pid1].add(pid2)
            #related_map[pid2].add(pid1)
    return score_map
    #return score_map, related_map

def extract_direct_links(df, all_paper_pairs=None):
    direct_score_map = {}
    paper_id_set = set(df["paperId"].dropna().unique())
    paperId_to_reference = load_paperId_lists(df, mode="reference")
    paperId_to_citation = load_paperId_lists(df, mode="citation")
    assert len(paperId_to_reference) > 0, "reference paperId list is empty"
    for pid, cited_set in paperId_to_reference.items():
        for cited_pid in cited_set:
            if cited_pid in paper_id_set:
                pair = tuple(sorted((pid, cited_pid)))
                direct_score_map[pair] = 1.0
    for pid, citing_set in paperId_to_citation.items():
        for citing_pid in citing_set:
            if citing_pid in paper_id_set:
                pair = tuple(sorted((pid, citing_pid)))
                direct_score_map[pair] = 1.0
    if all_paper_pairs:
        for pair in all_paper_pairs:
            if pair not in direct_score_map:
                direct_score_map[pair] = 0.0  ########
    return direct_score_map

def main():
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} rows from citation file.")
    paperId_to_ids = load_paperId_lists(df, MODE)
    all_paper_ids = list(df["paperId"].dropna().unique())
    all_pairs = [tuple(sorted(p)) for p in combinations(all_paper_ids, 2)]
    #print(paperId_to_ids)

    print(f"Prepared paperId sets for {len(paperId_to_ids)} papers.")
    print("Computing overlap scores and thresholded relationships...")
    #score_map, related_map = compute_overlap_scores(paperId_to_ids)
    score_map = compute_overlap_scores(paperId_to_ids)

    print("Extracting direct citation relationships...")
    direct_map = extract_direct_links(df, all_paper_pairs=all_pairs)

    os.makedirs(os.path.dirname(OUTPUT_PICKLE_SCORE), exist_ok=True)
    with open(OUTPUT_PICKLE_SCORE, "wb") as f:
        pickle.dump(score_map, f)
    #with open(OUTPUT_PICKLE_THRESHOLD, "wb") as f:
    #    pickle.dump(related_map, f)
    with open(OUTPUT_PICKLE_DIRECT, "wb") as f:
        pickle.dump(direct_map, f)

    print("✅ Done. All data saved:")
    print(f"  - Overlap scores:      {OUTPUT_PICKLE_SCORE}")
    #print(f"  - Related paper pairs: {OUTPUT_PICKLE_THRESHOLD}")
    print(f"  - Direct citations:    {OUTPUT_PICKLE_DIRECT}")


if __name__ == "__main__":
    main()

"""
Loaded 4547 rows from citation file.
Prepared paperId sets for 4544 papers.
Computing overlap scores and thresholded relationships...
100%|█████████████| 10321696/10321696 [00:26<00:00, 384970.89it/s]
Extracting direct citation relationships...
✅ Done. All data saved:
  - Overlap scores:      data/processed/modelcard_citation_overlap_by_paperId_score.pickle
  - Related paper pairs: data/processed/modelcard_citation_overlap_by_paperId_related.pickle
  - Direct citations:    data/processed/modelcard_citation_direct_relation.pickle
"""