"""
Author: Zhengyuan Dong
Date: 2025-04-03
Description: Batch query Semantic Scholar using corpusId with API key support, rate limiting, citation and reference extraction, error filtering, and output enriched citation results to Parquet.
Usage: python -m src.data_gt.step3_API_query
"""

import pandas as pd
import os, json, time, requests
from tqdm import tqdm
from dotenv import load_dotenv

# Constants
INPUT_PARQUET = "data/processed/final_integration_with_paths.parquet"
OUTPUT_PARQUET = "data/processed/modelcard_citation_enriched.parquet"
BATCH_SIZE = 100
API_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "corpusId,paperId,title,authors,year,venue,citations,references"


def main():
    # Load API key from .env if available
    load_dotenv()
    API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    USE_API_KEY = API_KEY is not None

    # Adjust rate limit based on presence of API key
    DELAY_SECONDS = 1.0 if USE_API_KEY else 3.0

    # Load input data
    df_input = pd.read_parquet(INPUT_PARQUET)
    corpus_ids = df_input['corpusid'].dropna().unique().astype(int).tolist()

    results = []

    for i in tqdm(range(0, len(corpus_ids), BATCH_SIZE), desc="Batch querying Semantic Scholar"):
        batch_ids = corpus_ids[i:i + BATCH_SIZE]
        payload = {"ids": [f"CorpusID:{cid}" for cid in batch_ids]}
        params = {"fields": FIELDS}
        headers = {"Content-Type": "application/json"}
        if USE_API_KEY:
            headers["x-api-key"] = API_KEY

        start_time = time.time()
        try:
            response = requests.post(API_ENDPOINT, headers=headers, params=params, json=payload)
        except Exception as e:
            print(f"Request error: {e}")
            continue

        if response.status_code == 200:
            batch_results = response.json()
            for res in batch_results:
                if res is None or not isinstance(res, dict):
                    continue
                try:
                    original_response = json.dumps(res)
                    citing_papers = res.get('citations', [])
                    cited_papers = res.get('references', [])

                    parsed_response = json.dumps({
                        "citing_papers": citing_papers,
                        "cited_papers": cited_papers
                    })

                    results.append({
                        "corpusId": res.get('corpusId', ''),
                        "paperId": res.get('paperId', ''),
                        "title": res.get('title', ''),
                        "year": res.get('year', ''),
                        "venue": res.get('venue', ''),
                        "original_response": original_response,
                        "parsed_response": parsed_response
                    })
                except Exception as err:
                    print(f"[Warning] Skipping one result due to error: {err}")
        else:
            print(f"[Batch Error {response.status_code}] {response.text}")

        elapsed_time = time.time() - start_time
        wait_time = max(0, DELAY_SECONDS - elapsed_time)
        if wait_time > 0:
            time.sleep(wait_time)

    # Save results to Parquet
    df_results = pd.DataFrame(results)
    df_results.to_parquet(OUTPUT_PARQUET, compression='zstd', engine='pyarrow', index=False)
    print(f"✅ Data saved to {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
    print("Script executed successfully.")

"""
Batch querying Semantic Scholar: 100%|█| 46/46 [05:17<00:00,  6.90
✅ Data saved to data/processed/modelcard_citation_enriched.parquet
Script executed successfully.
"""