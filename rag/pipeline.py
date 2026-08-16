from indexes.db import get_connection
from rag.llm import generate
from rag.prompts import build_prompt
from rag.query_rewriter import rewrite_query
from rag.reranker import rerank
from rag.retriever import RETRIEVERS


def answer(
    question,
    method="hybrid",
    variant="flexible",
    num_results=5,
    conn=None,
    use_query_rewriting=False,
    use_reranking=True,
):
    conn = conn or get_connection()
    retriever = RETRIEVERS[method]

    search_query = rewrite_query(question) if use_query_rewriting else question

    # retrieve a larger pool when reranking, since the point is to let the
    # reranker pick the best num_results out of more candidates than plain
    # retrieval would return
    candidate_pool = num_results * 2 if use_reranking else num_results
    chunks = retriever(conn, search_query, num_results=candidate_pool)

    # rerank against the user's original question, not the rewritten
    # search query — relevance should match what they actually asked
    if use_reranking:
        chunks = rerank(question, chunks, top_k=num_results)

    messages = build_prompt(question, chunks, variant=variant)
    response_text = generate(messages)

    return {
        "answer": response_text,
        "search_query": search_query,
        "sources": [
            {"title": c["title"], "heading": c["heading"], "url": c["url"]}
            for c in chunks
        ],
    }


if __name__ == "__main__":
    result = answer("How much do international students pay for undergraduate study?")

    print(f"search query: {result['search_query']!r}")
    print()
    print(result["answer"])
    print()
    print("Sources:")
    for s in result["sources"]:
        print(f"- {s['title']} — {s['heading']} ({s['url']})")
