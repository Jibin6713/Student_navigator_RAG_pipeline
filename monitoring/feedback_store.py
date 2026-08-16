def init_feedback_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            method TEXT NOT NULL,
            variant TEXT NOT NULL,
            rating INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


def save_feedback(conn, question, answer, rating, method="hybrid", variant="flexible"):
    conn.execute(
        """
        INSERT INTO feedback (question, answer, method, variant, rating)
        VALUES (?, ?, ?, ?, ?)
        """,
        (question, answer, method, variant, rating),
    )
    conn.commit()
