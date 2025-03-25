import sqlite3
import time
import threading
import random
import json

DATABASE_FILE = "paper_index_mini.db"

# Load titles from file
with open("tmp_dedup_titles.json", "r", encoding="utf-8") as f:
    dedup_titles = json.load(f)
test_titles = random.choices(dedup_titles, k=100)

def query_single(title, db_path=DATABASE_FILE):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT corpusid FROM papers WHERE title = ?", (title.lower(),))
    result = cur.fetchone()
    conn.close()
    return result

def test_single_thread(titles):
    results = []  ########
    start = time.time()
    for title in titles:
        res = query_single(title)
        results.append({"title": title, "corpusid": res[0] if res else None})  ########
    end = time.time()
    print(f"⏱️ Single-threaded: {len(titles)} queries took {end - start:.2f}s ({len(titles)/(end - start):.2f} queries/sec)")
    print("🔍 Sample results (first 5):", results[:5])  ########
    return results  ########

def thread_worker(titles, output, idx):
    local_results = []
    for title in titles:
        res = query_single(title)
        local_results.append({"title": title, "corpusid": res[0] if res else None})  ########
    output[idx] = local_results

def test_multi_thread(titles, num_threads=4):
    threads = []
    results = [None] * num_threads
    batch_size = len(titles) // num_threads
    start = time.time()
    for i in range(num_threads):
        batch = titles[i*batch_size : (i+1)*batch_size]
        t = threading.Thread(target=thread_worker, args=(batch, results, i))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    end = time.time()
    all_results = [item for sublist in results if sublist for item in sublist]  ########
    print(f"⚡ Multi-threaded ({num_threads} threads): {len(titles)} queries took {end - start:.2f}s ({len(titles)/(end - start):.2f} queries/sec)")
    print("🔍 Sample results (first 5):", all_results[:5])  ########
    return all_results  ########

if __name__ == "__main__":
    random.shuffle(test_titles)
    
    # Single-threaded test
    single_results = test_single_thread(test_titles)
    
    # Multi-threaded test
    multi_results = test_multi_thread(test_titles, num_threads=4)
    
    # Save to JSON
    with open("processed_results.json", "w", encoding="utf-8") as f:  ########
        json.dump(multi_results, f, indent=2, ensure_ascii=False)  ########
    print(f"✅ All results saved to processed_results.json ({len(multi_results)} entries)")  ########
