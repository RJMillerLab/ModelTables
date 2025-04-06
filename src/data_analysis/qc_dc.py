import os
import json

unique_files_path = "data/deduped/unique_files.txt"
stats_path = "data/deduped/stats.json"
duplicate_groups_path = "data/deduped/duplicate_groups.json"
valid_files_second_path = "data/analysis/valid_file_list_github.txt"

with open(stats_path, "r") as f:
    stats = json.load(f)

def load_file_list(file_path):
    with open(file_path, "r") as f:
        return set(line.strip() for line in f if line.strip())

unique_files = load_file_list(unique_files_path)
valid_files_github_first = stats["cross_unique_files"]["github"]
valid_files_github_first_set = set(valid_files_github_first)
assert len(unique_files) == stats["cross_unique_counts"]["github"] + stats["cross_unique_counts"]["hugging"] + stats["cross_unique_counts"]["html"] + stats["cross_unique_counts"]["llm"]
valid_files_second = load_file_list(valid_files_second_path)

extra_files = valid_files_second - valid_files_github_first_set

print("Stats for unique files:")
print("Hugging:", stats["cross_unique_counts"]["hugging"])
print("Github:", stats["cross_unique_counts"]["github"])
print("HTML:", stats["cross_unique_counts"]["html"])
print("LLM:", stats["cross_unique_counts"]["llm"])

print("Number of unique files in first code (github):", len(valid_files_github_first_set))
print("Number of valid files in second code:", len(valid_files_second))
valid_files_second_set = list(set(valid_files_second))
print("Number of deduplicated valid files in second code:", len(valid_files_second_set))
print("Extra files in second code compared to first:", len(extra_files))
print("\nList of extra files:")
for f in extra_files:
    print(f)

with open(duplicate_groups_path, "r") as f:
    duplicate_groups = json.load(f)

extra_details = []
for group in duplicate_groups:
    canonical = group.get("canonical")
    duplicates = group.get("duplicates", [])
    if canonical in extra_files or any(d in extra_files for d in duplicates):
        extra_details.append(group)

print("\nExtra file's duplicate group:")
for group in extra_details:
    print(json.dumps(group, indent=2))
