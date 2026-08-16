from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT/"data"/"chunks"/"chunks.jsonl"


def load_chunks():
    chunks = []


    with open(CHUNKS_FILE,"r",encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks        