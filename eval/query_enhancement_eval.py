import json
import random

from eval.retrieval_eval import PROJECT_ROOT, evaluate_method, load_ground_truth
from rag.query_rewriter import rewrite_query
from rag.reranker import rerank
from rag.retriever import hybrid_search

RESULTS_PATH = PROJECT_ROOT / "data" / "eval" / "query_enhancement_eval_results.json"


def make_retriever(use_query_rewriting, use_reranking):
    def retriever(conn, question, num_results=5):
        search_query = rewrite_query(question) if use_query_rewriting else question

        pool = num_results * 2 if use_reranking else num_results
        chunks = hybrid_search(conn, search_query, num_results=pool)

        if use_reranking:
            chunks = rerank(question, chunks, top_k=num_results)

        return chunks

    return retriever


CONFIGURATIONS = {
    "hybrid": make_retriever(use_query_rewriting=False, use_reranking=False),
    "hybrid+rewrite": make_retriever(use_query_rewriting=True, use_reranking=False),
    "hybrid+rerank": make_retriever(use_query_rewriting=False, use_reranking=True),
    "hybrid+rewrite+rerank": make_retriever(use_query_rewriting=True, use_reranking=True),
}


def run_evaluation(sample_size=100, seed=42, num_results=5):
    ground_truth = load_ground_truth()

    rng = random.Random(seed)
    sample = rng.sample(ground_truth, min(sample_size, len(ground_truth)))

    results = []

    for name, retriever in CONFIGURATIONS.items():
        print(f"Evaluating: {name}")
        result = evaluate_method(name, retriever, sample, num_results=num_results)
        print(f"  hit_rate={result['hit_rate']:.3f}  mrr={result['mrr']:.3f}")
        results.append(result)

    return results


def save_results(results):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def print_results(results):
    print()
    print(f"{'configuration':<24} {'hit_rate':>10} {'mrr':>10}")

    for r in results:
        print(f"{r['method']:<24} {r['hit_rate']:>10.3f} {r['mrr']:>10.3f}")


if __name__ == "__main__":
    results = run_evaluation()
    save_results(results)
    print_results(results)
