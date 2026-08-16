from pydantic import BaseModel

from rag.llm import generate_structured

RERANK_INSTRUCTIONS = """
You are ranking search results for a University of Auckland student
information assistant. Given a QUESTION and a numbered list of candidate
passages, return the passage numbers ordered from most to least relevant
to answering the question. Include every number exactly once.
""".strip()


class RankedResults(BaseModel):
    ranked_indices: list[int]


def rerank(question, chunks, top_k=5):
    if not chunks:
        return chunks

    candidates = "\n\n".join(
        f"[{i}] {c['heading']}\n{c['text']}" for i, c in enumerate(chunks)
    )
    user_prompt = f"QUESTION:\n{question}\n\nCANDIDATES:\n{candidates}"

    messages = [
        {"role": "system", "content": RERANK_INSTRUCTIONS},
        {"role": "user", "content": user_prompt},
    ]

    result = generate_structured(messages, RankedResults)

    seen = set()
    ranked = []

    for i in result.ranked_indices:
        if 0 <= i < len(chunks) and i not in seen:
            ranked.append(chunks[i])
            seen.add(i)

    # safety net: append anything the model dropped, in original order
    ranked += [c for i, c in enumerate(chunks) if i not in seen]

    return ranked[:top_k]
