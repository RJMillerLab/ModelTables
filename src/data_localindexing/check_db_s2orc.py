import sqlite3

db_path = "paper_index_mini.db"

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("SELECT * FROM papers LIMIT 10;")
rows = cur.fetchall()

for row in rows:
    print(row)

conn.close()

