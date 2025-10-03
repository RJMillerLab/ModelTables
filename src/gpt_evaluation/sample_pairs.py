import os, json, random, pandas as pd
base = "data/processed/deduped_hugging_csvs_v2"
all_csv = []
for root, _, files in os.walk(base):
    for f in files:
        if f.lower().endswith(".csv"):
            all_csv.append(os.path.join(root, f))
random.seed(42)
K = 50  # 采样对数，可改
pairs = []
for _ in range(K):
    a, b = random.sample(all_csv, 2)
    def csv_to_md(p, n=10):
        try:
            df = pd.read_csv(p, nrows=n)
            return df.to_markdown(index=False)
        except Exception:
            return f"(failed to read) {p}"
    pairs.append({
        "id": f"pair-{len(pairs)+1}",
        "table_a_md": csv_to_md(a),
        "table_b_md": csv_to_md(b)
    })
os.makedirs("output", exist_ok=True)
out = "output/_tmp_table_pairs.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for r in pairs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(out)
