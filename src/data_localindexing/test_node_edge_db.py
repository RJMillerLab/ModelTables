import os
import glob
import argparse
import subprocess  # For optional debugging
import json  ######## checkpoint: used for saving progress
from tqdm import tqdm
import kuzu

DB_PATH = "demo_graph_db/"

def test_graph_contents():
    db = kuzu.Database(DB_PATH)
    conn = kuzu.Connection(db)

    # 打印总节点数
    result = conn.execute("MATCH (n:Corpus) RETURN COUNT(n) AS node_count;")
    total_nodes = result.get_next()[0]
    result.close()
    print("📊 Total Corpus nodes:", total_nodes)

    # 打印前10个节点
    print("🧩 Sample Corpus nodes:")
    result = conn.execute("MATCH (n:Corpus) RETURN n.id AS id LIMIT 10;")
    while result.has_next():
        rec = result.get_next()
        print("Node id:", rec[0])
    result.close()

    # 打印总边数
    result = conn.execute("MATCH ()-[r:Cites]->() RETURN COUNT(r) AS edge_count;")
    total_edges = result.get_next()[0]
    result.close()
    print("🔗 Total Cites edges:", total_edges)

    # 打印前10条边
    print("🔍 Sample Cites edges:")
    result = conn.execute("""
        MATCH (a:Corpus)-[r:Cites]->(b:Corpus)
        RETURN a.id AS citing, b.id AS cited, r.citationid, r.isinfluential, r.contexts, r.intents
        LIMIT 10;
    """)
    while result.has_next():
        rec = result.get_next()
        # 如果返回的是 tuple，可以直接打印，如果需要格式化可以做调整
        print("Edge:", rec)
    result.close()

    conn.close()
    db.close()

# 调用测试函数
test_graph_contents()

