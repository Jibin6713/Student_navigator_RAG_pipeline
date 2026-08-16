import sqlite3
from pathlib import Path

import sqlite_vec

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_DIR = PROJECT_ROOT / "data" / "index"
DB_PATH = DB_DIR / "knowledge_base.db"

DB_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_FIELDS = (
    "chunk_id", "url", "title", "heading",
    "section_number", "source", "text", "char_count",
)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    return conn


# rebuilt fresh from chunks.jsonl each run, rowids drive the fts/vec joins
def rebuild_chunks_table(conn, chunks):
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE NOT NULL,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            heading TEXT NOT NULL,
            section_number INTEGER NOT NULL,
            source TEXT NOT NULL,
            text TEXT NOT NULL,
            char_count INTEGER NOT NULL
        )
    """)

    conn.executemany(
        """
        INSERT INTO chunks
            (chunk_id, url, title, heading, section_number, source, text, char_count)
        VALUES
            (:chunk_id, :url, :title, :heading, :section_number, :source, :text, :char_count)
        """,
        chunks,
    )
    conn.commit()
