import json
import sqlite3

def save_chunks(chunks, path="chunks.sqlite"):
    """SQLite demonstrates metadata persistence and reproducible ingestion."""
    # The transaction commits all inserts together, or rolls back on failure.
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, body TEXT)")
        db.execute("DELETE FROM chunks")
        db.executemany("INSERT INTO chunks VALUES (?, ?)",
                       [(c["id"], json.dumps(c)) for c in chunks])

def load_chunks(path="chunks.sqlite"):
    with sqlite3.connect(path) as db:
        return [json.loads(row[0]) for row in db.execute("SELECT body FROM chunks ORDER BY id")]
