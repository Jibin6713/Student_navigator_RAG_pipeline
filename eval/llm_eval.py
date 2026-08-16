import json
import random
import threading
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from eval.evaluation_utils import calc_total_price, llm_structured_retry, map_progress
from eval.retrieval_eval import load_ground_truth
from indexes.db import get_connection
from rag.llm import client, generate
from rag.prompts import PROMPTS, build_prompt
from rag.retriever import hybrid_search

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "data" / "eval" / "llm_eval_results.jsonl"

JUDGE_MODEL = "gpt-4.1-mini"
NUM_RESULTS = 5

# retrieval is held fixed at "hybrid" (the winner from retrieval_eval.py) so
# this isolates the effect of the prompt variant on generation quality

_thread_local = threading.local()


def get_thread_connection():
    if not hasattr(_thread_local, "conn"):
        _thread_local.conn = get_connection()

    return _thread_local.conn


JUDGE_INSTRUCTIONS = """
You are evaluating answers from a University of Auckland student Q&A
assistant. You will be given a QUESTION, the CONTEXT the assistant had
access to, and the assistant's ANSWER.

Judge two independent things:
- relevance: does the answer actually address the question asked?
- faithfulness: is the answer fully supported by the CONTEXT, without
  introducing any fact that is not present in the CONTEXT?
""".strip()

JUDGE_PROMPT_TEMPLATE = """
QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
""".strip()


class JudgeScore(BaseModel):
    relevance: Literal["RELEVANT", "PARTLY_RELEVANT", "NOT_RELEVANT"]
    faithfulness: Literal["FAITHFUL", "PARTLY_FAITHFUL", "NOT_FAITHFUL"]
    explanation: str


def judge_answer(question, context, answer):
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, context=context, answer=answer
    )
    return llm_structured_retry(
        client, JUDGE_INSTRUCTIONS, prompt, JudgeScore, JUDGE_MODEL
    )


def evaluate_question(record, variant):
    conn = get_thread_connection()

    chunks = hybrid_search(conn, record["question"], num_results=NUM_RESULTS)
    messages = build_prompt(record["question"], chunks, variant=variant)
    answer = generate(messages)

    context = "\n\n".join(c["text"] for c in chunks)
    judge_result, judge_usage = judge_answer(record["question"], context, answer)

    result = {
        "question": record["question"],
        "variant": variant,
        "answer": answer,
        "relevance": judge_result.relevance,
        "faithfulness": judge_result.faithfulness,
        "explanation": judge_result.explanation,
    }

    return result, judge_usage


def process_task(task):
    record, variant = task

    try:
        return evaluate_question(record, variant)
    except Exception as e:
        print(f"skipped ({variant}): {e}")
        return None, None


def run_llm_eval(sample_size=100, seed=42, ground_truth=None):
    ground_truth = ground_truth if ground_truth is not None else load_ground_truth()

    rng = random.Random(seed)
    sample = rng.sample(ground_truth, min(sample_size, len(ground_truth)))

    tasks = [(record, variant) for record in sample for variant in PROMPTS]
    results = map_progress(process_task, tasks, max_workers=6)

    records = []
    usages = []

    for record, usage in results:
        if record is not None:
            records.append(record)
        if usage is not None:
            usages.append(usage)

    return records, usages


def save_results(records):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records):
    summary = defaultdict(lambda: defaultdict(int))

    for r in records:
        counts = summary[r["variant"]]
        counts["total"] += 1
        counts[f"relevance:{r['relevance']}"] += 1
        counts[f"faithfulness:{r['faithfulness']}"] += 1

    return summary


def print_summary(summary):
    for variant, counts in summary.items():
        total = counts["total"]
        print(f"\n{variant} (n={total})")

        for key, value in sorted(counts.items()):
            if key == "total":
                continue
            print(f"  {key}: {value} ({value / total:.1%})")


if __name__ == "__main__":
    records, usages = run_llm_eval()
    save_results(records)

    print()
    print(f"Wrote {len(records)} results to {RESULTS_PATH}")
    print(f"Estimated judge cost: ${calc_total_price(usages):.4f} (generation cost not included)")

    print_summary(summarize(records))
