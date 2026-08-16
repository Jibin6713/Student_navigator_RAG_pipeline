from indexes.build_vector_index import search as vector_search
from indexes.keyword_index import search as keyword_search

# standard RRF constant — dampens the impact of any single rank position
RRF_K = 60


# combine two ranked lists by rank position, not raw score, since bm25
# and vector distance live on incompatible scales
def hybrid_search(conn, query, num_results=5, candidate_pool=None):
    pool_size = candidate_pool or num_results * 2

    keyword_results = keyword_search(conn, query, num_results=pool_size)
    vec_results = vector_search(conn, query, num_results=pool_size)

    scores = {}
    chunks_by_id = {}

    for results in (keyword_results, vec_results):
        for rank, chunk in enumerate(results):
            scores[chunk["chunk_id"]] = (
                scores.get(chunk["chunk_id"], 0.0) + 1.0 / (RRF_K + rank + 1)
            )
            chunks_by_id[chunk["chunk_id"]] = chunk

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:num_results]

    results = []
    for chunk_id in ranked_ids:
        chunk = dict(chunks_by_id[chunk_id])
        chunk["rrf_score"] = scores[chunk_id]
        results.append(chunk)

    return results


# common interface for the retrieval evaluation to loop over
RETRIEVERS = {
    "keyword": keyword_search,
    "vector": vector_search,
    "hybrid": hybrid_search,
}


if __name__ == "__main__":
    from indexes.db import get_connection

    conn = get_connection()
    query = "how much do international students pay"

    for name, retriever in RETRIEVERS.items():
        print(f"--- {name} ---")
        for r in retriever(conn, query, num_results=3):
            print(r["heading"], "|", r["url"])
        print()
