from rag.llm import generate

REWRITE_INSTRUCTIONS = """
You rewrite user questions into clear, specific search queries for a
University of Auckland student information search engine. Resolve vague
phrasing, expand abbreviations, and make implicit topics explicit (e.g.
"cost" -> "tuition fees"), but do not invent details the user didn't ask
about, and do not answer the question. Return ONLY the rewritten query
text, nothing else — no quotes, no explanation.
""".strip()


def rewrite_query(question):
    messages = [
        {"role": "system", "content": REWRITE_INSTRUCTIONS},
        {"role": "user", "content": question},
    ]

    return generate(messages).strip()


if __name__ == "__main__":
    for q in [
        "how much for international students",
        "when do i need to apply by for semester 2",
        "can i get money to help pay for uni",
    ]:
        print(f"{q!r} -> {rewrite_query(q)!r}")
