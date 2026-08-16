import time
from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

# verify against https://openai.com/api/pricing/ before trusting cost output
INPUT_PRICE_PER_MILLION = 0.40
OUTPUT_PRICE_PER_MILLION = 1.60


def calc_price(usage):
    input_cost = (usage.input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (usage.output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def calc_total_price(usages):
    return sum(calc_price(usage)["total_cost"] for usage in usages)


def llm_structured(client, instructions, user_prompt, output_type, model):
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]

    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=output_type,
    )

    return response.output_parsed, response.usage


def llm_structured_retry(
    client, instructions, user_prompt, output_type, model, max_retries=3
):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client, instructions, user_prompt, output_type, model
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


# runs f over seq concurrently (I/O-bound LLM calls benefit from threads
# despite the GIL); progress updates as futures actually complete, not
# in submission order
def map_progress(f, seq, max_workers=6):
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        with tqdm(total=len(seq)) as progress:
            futures = []

            for item in seq:
                future = pool.submit(f, item)
                future.add_done_callback(lambda _: progress.update())
                futures.append(future)

            for future in futures:
                results.append(future.result())

    return results
