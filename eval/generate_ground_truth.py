import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from eval.evaluation_utils import calc_total_price, llm_structured_retry, map_progress
from indexes.load_chunks import load_chunks

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "eval" / "ground_truth.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4.1-mini"
QUESTIONS_PER_CHUNK = 3

client = OpenAI()

INSTRUCTIONS = """
You are generating evaluation questions for a University of Auckland student
Q&A assistant. Given one section from the university's website, write
realistic questions a prospective or current student might ask that this
section's content answers. Questions must be answerable using ONLY the given
text, and should not mention "the text" or "this section" — write them the
way a student naturally would ask.
""".strip()

USER_PROMPT_TEMPLATE = """
TITLE: {title}
SECTION: {heading}
CONTENT:
{text}

Write {n} questions.
""".strip()


class GeneratedQuestions(BaseModel):
    questions: list[str]


def generate_questions(chunk, n=QUESTIONS_PER_CHUNK):
    user_prompt = USER_PROMPT_TEMPLATE.format(
        title=chunk["title"], heading=chunk["heading"], text=chunk["text"], n=n
    )

    result, usage = llm_structured_retry(
        client, INSTRUCTIONS, user_prompt, GeneratedQuestions, MODEL
    )

    return result.questions, usage


def process_chunk(chunk):
    try:
        questions, usage = generate_questions(chunk)
    except Exception as e:
        print(f"skipped {chunk['chunk_id']}: {e}")
        return [], None

    records = [
        {"question": question, "chunk_id": chunk["chunk_id"], "url": chunk["url"]}
        for question in questions
    ]

    return records, usage


def build_ground_truth(chunks):
    results = map_progress(process_chunk, chunks, max_workers=6)

    records = []
    usages = []

    for chunk_records, usage in results:
        records.extend(chunk_records)
        if usage is not None:
            usages.append(usage)

    return records, usages


def save_ground_truth(records):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    chunks = load_chunks()
    records, usages = build_ground_truth(chunks)
    save_ground_truth(records)

    print()
    print(f"Wrote {len(records)} ground-truth questions to {OUTPUT_PATH}")
    print(f"Estimated cost: ${calc_total_price(usages):.4f}")
