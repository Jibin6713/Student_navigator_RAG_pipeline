import re

from indexes.db import get_connection, rebuild_chunks_table
from indexes.load_chunks import load_chunks


# FTS5 parses the MATCH string as its own query syntax (AND/OR/NOT, quotes,
# column filters) — quoting each token individually makes punctuation in
# arbitrary user queries safe instead of raising a syntax error.
# joined with OR (not FTS5's default implicit AND) since natural-language
# questions are full of filler words ("what", "the", "are") unlikely to all
# appear in one chunk — bm25() still ranks rows with more/rarer matches higher
def sanitize_fts_query(query):
    tokens = re.findall(r"\w+", query)
    return " OR ".join(f'"{token}"' for token in tokens)


def build_fts_index(conn):
    conn.execute("DROP TABLE IF EXISTS chunks_fts")
    conn.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            title,
            heading,
            text,
            content='chunks',
            content_rowid='id'
        )
    """)
    conn.execute("""
        INSERT INTO chunks_fts (rowid, chunk_id, title, heading, text)
        SELECT id, chunk_id, title, heading, text FROM chunks
    """)
    conn.commit()


def build():
    conn = get_connection()
    rebuild_chunks_table(conn, load_chunks())
    build_fts_index(conn)
    return conn


def search(conn, query, num_results=5):
    rows = conn.execute(
        """
        SELECT c.chunk_id, c.url, c.title, c.heading, c.text,
               bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (sanitize_fts_query(query), num_results),
    ).fetchall()

    return [dict(row) for row in rows]


if __name__ == "__main__":
    conn = build()

    results = search(conn, "tuition fees for international students")

    for r in results:
        print(f"{r['score']:.3f}  {r['heading']}  |  {r['url']}")
