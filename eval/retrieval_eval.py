import json
import threading
from pathlib import Path

from eval.evaluation_utils import map_progress
from indexes.db import get_connection
from rag.retriever import RETRIEVERS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "eval" / "ground_truth.jsonl"
RESULTS_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_results.json"

NUM_RESULTS = 5

# sqlite3 connections can't cross threads — cache one per worker thread
# instead of sharing a single connection across the pool
_thread_local = threading.local()


def get_thread_connection():
    if not hasattr(_thread_local, "conn"):
        _thread_local.conn = get_connection()

    return _thread_local.conn


def load_ground_truth():
    records = []

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def rank_of_target(results, target_chunk_id):
    for i, result in enumerate(results, start=1):
        if result["chunk_id"] == target_chunk_id:
            return i

    return None


def hit_rate(ranks):
    return sum(1 for rank in ranks if rank is not None) / len(ranks)


def mrr(ranks):
    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)


def evaluate_method(method_name, retriever, ground_truth, num_results=NUM_RESULTS):
    def score_one(record):
        conn = get_thread_connection()
        results = retriever(conn, record["question"], num_results=num_results)
        return rank_of_target(results, record["chunk_id"])

    ranks = map_progress(score_one, ground_truth, max_workers=6)

    return {
        "method": method_name,
        "hit_rate": hit_rate(ranks),
        "mrr": mrr(ranks),
    }


def run_evaluation(ground_truth=None, num_results=NUM_RESULTS):
    ground_truth = ground_truth if ground_truth is not None else load_ground_truth()

    results = []

    for method_name, retriever in RETRIEVERS.items():
        print(f"Evaluating: {method_name}")
        result = evaluate_method(method_name, retriever, ground_truth, num_results)
        print(f"  hit_rate={result['hit_rate']:.3f}  mrr={result['mrr']:.3f}")
        results.append(result)

    return results


def save_results(results):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def print_results(results):
    print()
    print(f"{'method':<10} {'hit_rate':>10} {'mrr':>10}")

    for r in results:
        print(f"{r['method']:<10} {r['hit_rate']:>10.3f} {r['mrr']:>10.3f}")


if __name__ == "__main__":
    results = run_evaluation()
    save_results(results)
    print_results(results)
