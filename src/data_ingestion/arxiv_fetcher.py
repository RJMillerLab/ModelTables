"""
Author: Zhengyuan Dong
Date: 2025-02-15
Description: arxiv fetcher by arxiv API
"""

import os
import json
import arxiv
from tqdm import tqdm

RESULTS_FILE = "data/processed/arxiv_results.json"
TMP_TEST_FILE = "data/tmp/arxiv_test.json"

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
os.makedirs(os.path.dirname(TMP_TEST_FILE), exist_ok=True)

def search_arxiv(title_query, max_results=5):
    search = arxiv.Search(
        query=f"ti:{title_query}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    results = []
    for result in search.results():
        results.append({
            "title": result.title,
            "pdf_url": result.pdf_url,
            "source_available": "source" in result.links[-1].href if result.links else False
        })
    return results

def batch_search_arxiv(titles):
    all_results = {}
    for title in tqdm(titles, desc="Searching arXiv"):
        all_results[title] = search_arxiv(title)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"✅ Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    test_title = "BERT: Pre-training of Deep Bidirectional Transformers"
    test_results = search_arxiv(test_title)
    
    test_titles = [
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Attention Is All You Need"
    ]
    
    batch_search_arxiv(test_titles)

    with open(TMP_TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=4)

    print(f"🔍 Test results saved to {TMP_TEST_FILE}")
