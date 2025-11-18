import json
from collections import defaultdict

# 1. 加载文件
with open('data/processed/modelcard_dedup_titles.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 统计基本信息
print(f"总数据量: {len(data)} 条记录")

# 3. 分析数据结构（如果是列表套字典的结构）
if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
    structure_stats = defaultdict(int)
    for item in data:
        structure_stats[str(item.keys())] += 1
    
    print("\n数据结构统计:")
    for structure, count in structure_stats.items():
        print(f"- {count} 条记录具有结构: {structure}")

# 4. 如果是字典结构
elif isinstance(data, dict):
    print("\n字典结构键名:", data.keys())
    print("字典值类型统计:")
    for k, v in data.items():
        print(f"- '{k}': {type(v).__name__}")
