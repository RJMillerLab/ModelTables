import sqlite3

db_path = "citation_memory.db"

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("SELECT * FROM citations LIMIT 10;")
rows = cur.fetchall()

for row in rows:
    print(row)

conn.close()

