from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4.1-mini"

client = OpenAI()


# takes the messages list already built by prompts.build_prompt() —
# this function owns nothing about prompt construction, just the API call
def generate(messages, model=MODEL):
    response = client.responses.create(
        model=model,
        input=messages,
    )

    return response.output_text


def generate_structured(messages, output_type, model=MODEL):
    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=output_type,
    )

    return response.output_parsed


if __name__ == "__main__":
    from rag.prompts import build_prompt

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

    print(generate(messages))
