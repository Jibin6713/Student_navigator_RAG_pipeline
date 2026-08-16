import time

import sqlite_vec
from dotenv import load_dotenv
from openai import OpenAI

from indexes.db import get_connection, rebuild_chunks_table
from indexes.load_chunks import load_chunks

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100

client = OpenAI()


def embed_batch(batch, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def embed_texts(texts):
    embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = embed_batch(batch)
        embeddings.extend(item.embedding for item in response.data)

    return embeddings


def build_vec_index(conn, chunks):
    conn.execute("DROP TABLE IF EXISTS chunks_vec")
    conn.execute(f"""
        CREATE VIRTUAL TABLE chunks_vec USING vec0(
            embedding float[{EMBEDDING_DIM}]
        )
    """)

    embeddings = embed_texts([chunk["text"] for chunk in chunks])

    conn.executemany(
        "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
        [
            (i + 1, sqlite_vec.serialize_float32(embedding))
            for i, embedding in enumerate(embeddings)
        ],
    )
    conn.commit()


def build():
    conn = get_connection()
    chunks = load_chunks()
    rebuild_chunks_table(conn, chunks)
    build_vec_index(conn, chunks)
    return conn


def search(conn, query, num_results=5):
    query_embedding = embed_texts([query])[0]

    rows = conn.execute(
        """
        SELECT c.chunk_id, c.url, c.title, c.heading, c.text,
               v.distance
        FROM chunks_vec v
        JOIN chunks c ON c.id = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (sqlite_vec.serialize_float32(query_embedding), num_results),
    ).fetchall()

    return [dict(row) for row in rows]


if __name__ == "__main__":
    conn = build()

    results = search(conn, "how much do international students pay")

    for r in results:
        print(f"{r['distance']:.3f}  {r['heading']}  |  {r['url']}")
