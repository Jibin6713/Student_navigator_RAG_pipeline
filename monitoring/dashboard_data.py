import json
from pathlib import Path

from indexes.db import get_connection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "llm_eval_results.jsonl"
RETRIEVAL_EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_results.json"


def load_feedback():
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT question, answer, method, variant, rating, created_at
        FROM feedback
        ORDER BY created_at
        """
    ).fetchall()

    return [dict(row) for row in rows]


def load_llm_eval_results():
    if not LLM_EVAL_PATH.exists():
        return []

    records = []

    with open(LLM_EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def load_retrieval_eval_results():
    if not RETRIEVAL_EVAL_PATH.exists():
        return []

    with open(RETRIEVAL_EVAL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
