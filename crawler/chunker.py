import json
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import NavigableString

## configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_HTML_DIR = PROJECT_ROOT / "data" / "raw_html"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

MIN_CHUNK_CHARS = 30
MAX_CHUNK_CHARS = 1500

INTRO_HEADING = "Overview"


# walk the DOM, starting a new section every time an h2/h3 is hit
def extract_sections(main):
    sections = []
    state = {"heading": INTRO_HEADING, "buffer": []}

    def flush():
        text = "\n".join(part for part in state["buffer"] if part)
        if text.strip():
            sections.append({"heading": state["heading"], "text": text})
        state["buffer"] = []

    def walk(node):
        for child in node.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    state["buffer"].append(text)
            elif child.name in ("h2", "h3"):
                flush()
                state["heading"] = child.get_text(strip=True) or state["heading"]
            else:
                walk(child)

    walk(main)
    flush()

    return sections


# keep chunks from growing too large for embedding/retrieval later
def split_oversized(section):
    text = section["text"]

    if len(text) <= MAX_CHUNK_CHARS:
        return [section]

    parts = []
    lines = text.split("\n")
    current = []
    current_len = 0

    for line in lines:
        if current_len + len(line) > MAX_CHUNK_CHARS and current:
            parts.append("\n".join(current))
            current = []
            current_len = 0

        current.append(line)
        current_len += len(line) + 1

    if current:
        parts.append("\n".join(current))

    total = len(parts)
    return [
        {"heading": f"{section['heading']} ({i + 1}/{total})", "text": part}
        for i, part in enumerate(parts)
    ]


def chunk_page(cleaned_path: Path):
    with open(cleaned_path, "r", encoding="utf-8") as f:
        document = json.load(f)

    html_path = RAW_HTML_DIR / f"{cleaned_path.stem}.html"

    if not html_path.exists():
        print(f"No raw HTML for {cleaned_path.name}, skipping.")
        return []

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")
    main = soup.find("main", id="main")

    if main is None:
        print(f"No <main id='main'> in {html_path.name}, skipping.")
        return []

    for tag in main(["script", "style", "noscript"]):
        tag.decompose()

    sections = extract_sections(main)

    chunks = []

    for section in sections:
        if len(section["text"]) < MIN_CHUNK_CHARS:
            continue

        for part in split_oversized(section):
            section_number = len(chunks)
            chunks.append({
                "chunk_id": f"{cleaned_path.stem}::{section_number}",
                "url": document["url"],
                "title": document["title"],
                "heading": part["heading"],
                "section_number": section_number,
                "source": cleaned_path.stem,
                "text": part["text"],
                "char_count": len(part["text"]),
            })

    return chunks


def chunk_all():
    cleaned_files = sorted(CLEANED_DIR.glob("*.json"))

    all_chunks = []

    for cleaned_path in cleaned_files:
        chunks = chunk_page(cleaned_path)
        all_chunks.extend(chunks)

    output_path = CHUNKS_DIR / "chunks.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print()
    print("=" * 40)
    print(f"Pages processed : {len(cleaned_files)}")
    print(f"Chunks written  : {len(all_chunks)}")
    print(f"Output          : {output_path}")
    print("=" * 40)


if __name__ == "__main__":
    chunk_all()
