SYSTEM_PROMPT_STRICT = """
You are the UoA Student Navigator, an assistant that answers questions about
the University of Auckland using only the CONTEXT provided below.

Rules:
- Only use information found in the CONTEXT. Do not use outside knowledge.
- If the answer is not in the CONTEXT, say you don't know and suggest the
  user check the relevant University of Auckland webpage.
- Always cite the source URL(s) you used, at the end of your answer.
- Be concise and factual — this is used for things like fees, deadlines and
  entry requirements, where precision matters.
""".strip()

SYSTEM_PROMPT_FLEXIBLE = """
You are the UoA Student Navigator, an assistant that helps prospective and
current University of Auckland students.

Rules:
- Prefer information from the CONTEXT provided below — it is the most
  up-to-date and specific source available.
- You may use general knowledge to explain background concepts (e.g. what a
  GPA is), but any specific fact (fees, dates, requirements) must come from
  the CONTEXT.
- If the CONTEXT does not cover the question, say so plainly and suggest the
  user check the relevant University of Auckland webpage.
- Always cite the source URL(s) you used, at the end of your answer.
""".strip()

# kept as a registry so the LLM evaluation can loop over variants by name
PROMPTS = {
    "strict": SYSTEM_PROMPT_STRICT,
    "flexible": SYSTEM_PROMPT_FLEXIBLE,
}


def format_context(chunks):
    if not chunks:
        return "No relevant information was found."

    blocks = []

    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] {chunk['title']} — {chunk['heading']}\n"
            f"URL: {chunk['url']}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(blocks)


def build_prompt(question, chunks, variant="strict"):
    system_prompt = PROMPTS[variant]
    context = format_context(chunks)

    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


if __name__ == "__main__":
    sample_chunks = [
        {
            "title": "International student fees",
            "heading": "Undergraduate fees for international students",
            "url": "https://www.auckland.ac.nz/en/study/fees-and-money-matters/tuition-fees/international-student-fees.html",
            "text": "International undergraduate fees range from NZ$35,000 to NZ$45,000 per year depending on programme.",
        }
    ]

    messages = build_prompt(
        "How much do international students pay for undergraduate study?",
        sample_chunks,
        variant="strict",
    )

    for message in messages:
        print(f"--- {message['role']} ---")
        print(message["content"])
        print()
