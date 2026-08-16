# UoA Student Navigator

A Retrieval-Augmented Generation (RAG) assistant that answers questions about
studying at the **University of Auckland** — fees, entry requirements, application
deadlines, student support, campus life — using content crawled directly from
the university's own website, not a general-purpose LLM guessing from memory.

Built as a project for the [DataTalks.Club LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp).

> This README assumes no prior knowledge of the course. Every tool and technique
> used is explained here, including the ones not covered in the course itself.

---

## Table of contents

- [The problem](#the-problem)
- [The data](#the-data)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Example usage](#example-usage)
- [Evaluation](#evaluation)
- [Monitoring](#monitoring)
- [Evaluation criteria checklist](#evaluation-criteria-checklist)
- [Known limitations / what's left](#known-limitations--whats-left)

---

## The problem

The University of Auckland's website is large, deeply nested, and split across
dozens of pages per topic (fees, entry requirements, visas, accommodation,
scholarships...). A prospective or current student who wants a specific answer
— "how much does an international undergraduate degree cost?", "when is the
Semester Two application deadline?" — usually has to click through several
pages to find it, and the answer is often buried in dense text.

**UoA Student Navigator** is a chat assistant that answers these questions
directly, in plain language, with a citation back to the exact page the answer
came from. It only answers from content actually scraped off the university's
website — it does not rely on the LLM's general knowledge, which reduces the
risk of stale or hallucinated answers about fees, dates, or requirements.

## The data

The knowledge base is built from the University of Auckland's public website
(`auckland.ac.nz`), crawled from four seed sections: **Study**, **Students**,
**On-campus**, and **News**. This is original, self-collected data — not the
DataTalks.Club FAQ documents.

| | |
|---|---|
| Pages crawled | 101 |
| Chunks indexed | 889 (heading-based sections, see below) |
| Source | `auckland.ac.nz/en/study/*`, `/students/*`, `/on-campus/*`, `/news.html` |

## How it works

```
                    ┌─────────────────────────────────────────────┐
                    │              INGESTION (offline)             │
                    │                                               │
  auckland.ac.nz →  │  crawl → clean → chunk → embed → index       │
                    └─────────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────────┐
                    │           QUERY TIME (per question)          │
                    │                                               │
  user question  →  │  retrieve (keyword + vector, fused)          │
                    │       → build prompt with retrieved context  │
                    │       → LLM generates cited answer           │
                    └─────────────────────────────────────────────┘
                                        │
                                        ▼
                         Streamlit chat UI  +  👍/👎 feedback
                                        │
                                        ▼
                            Monitoring dashboard
```

### 1. Ingestion

- **Crawl** ([`crawler/crawler.py`](crawler/crawler.py)) — a breadth-first crawler
  (`requests` + `BeautifulSoup`) starting from 4 seed URLs, restricted to
  university-owned paths, saving both raw HTML and a cleaned text extraction
  per page.
- **Chunk** ([`crawler/chunker.py`](crawler/chunker.py)) — instead of naive
  fixed-size text splitting, each page is re-parsed and split at its own
  `<h2>`/`<h3>` heading boundaries. The site's authors already organized each
  page into topical sections ("Entry requirements", "Fees", "Key dates"); reusing
  that structure gives topically coherent chunks for free, rather than
  statistically inferring topic boundaries. Oversized sections (>1500 chars)
  are further split on paragraph boundaries.
- **Index** ([`indexes/`](indexes/)) — every chunk is loaded into **two** indexes
  living in one SQLite file (`data/index/knowledge_base.db`):
  - a **keyword index** using SQLite's built-in **FTS5** full-text search
    extension (BM25 ranking) — not covered in the course, explained below.
  - a **vector index** using **[sqlite-vec](https://github.com/asg017/sqlite-vec)**,
    a SQLite extension for storing and searching embedding vectors — also not
    covered in the course, explained below. Embeddings are generated with
    OpenAI's `text-embedding-3-small`.

### 2. Retrieval

Three retrieval strategies are implemented and compared (see
[Evaluation](#evaluation)):

- **keyword** — SQLite FTS5, BM25-ranked
- **vector** — `sqlite-vec`, cosine-similarity nearest neighbours
- **hybrid** — both of the above combined via **Reciprocal Rank Fusion (RRF)**:
  results are merged by *rank position* (`score += 1 / (60 + rank)`) rather than
  raw score, since BM25 and vector distance live on incompatible scales. This
  is the retrieval bonus item ("hybrid search combining text and vector search").

Two further retrieval bonus techniques were built and evaluated
([`rag/query_rewriter.py`](rag/query_rewriter.py), [`rag/reranker.py`](rag/reranker.py)):

- **Query rewriting** — an LLM call rewrites the user's raw question into a
  clearer search query before retrieval (e.g. *"how much for international
  students"* → *"tuition fees for international students at the University of
  Auckland"*), while the *original* question is still what gets answered.
- **Re-ranking** — instead of retrieving exactly the final top-5, hybrid search
  retrieves a larger pool (top-10), then an LLM re-ranks that pool against the
  original question and keeps the best 5. An LLM re-ranker was used instead of
  a cross-encoder model to avoid pulling in `torch`/`sentence-transformers` —
  a much heavier dependency than anything else in this project.

Both are implemented as toggleable steps in [`rag/pipeline.py`](rag/pipeline.py)
and were evaluated against each other (see [Evaluation](#evaluation)) — **only
re-ranking is enabled by default**, since query rewriting measurably hurt
retrieval quality on the evaluation set, for reasons explained there.

### 3. Generation

The top-ranked chunks are inserted into a prompt template
([`rag/prompts.py`](rag/prompts.py)) and sent to an OpenAI chat model
(`gpt-4.1-mini`). Two prompt variants were built and evaluated:

- **strict** — must answer only from the given context, or say it doesn't know
- **flexible** — may use general knowledge for background explanation, but
  facts (fees, dates, requirements) must still come from the context

(`flexible` is the current default — see [Evaluation](#evaluation) for why.)

### 4. Interface

A [Streamlit](https://streamlit.io) chat app ([`app/streamlit_app.py`](app/streamlit_app.py))
— ask a question, get an answer with an expandable list of cited sources, and
rate the answer 👍/👎.

### 5. Monitoring

Feedback (👍/👎, plus the question/answer/method/variant) is logged to SQLite
([`monitoring/feedback_store.py`](monitoring/feedback_store.py)), and a second
Streamlit page ([`app/pages/1_Dashboard.py`](app/pages/1_Dashboard.py))
visualizes it alongside the offline evaluation results.

## Tech stack

| Purpose | Tool | Covered in course? |
|---|---|---|
| Crawling | `requests`, `BeautifulSoup`, `lxml` | — |
| Keyword search | **SQLite FTS5** | No — see below |
| Vector search | **sqlite-vec** | No — see below |
| Embeddings | OpenAI `text-embedding-3-small` | Yes |
| Generation & judging | OpenAI `gpt-4.1-mini` | Yes |
| Structured LLM output | Pydantic + OpenAI `responses.parse` | No — see below |
| Interface | Streamlit | Yes (listed as an accepted option) |
| Dashboard charts | Plotly + pandas | — |
| Dependency management | [uv](https://docs.astral.sh/uv/) | — |

**Why SQLite FTS5 + sqlite-vec instead of the course's `minsearch` / a hosted
vector DB?** The project started with `minsearch` (the course's toy in-memory
search library), but it has no persistence — it rebuilds from scratch on every
process start and doesn't produce real BM25 scores. FTS5 is SQLite's built-in
full-text search extension: it stores an inverted index on disk, ranks results
with the industry-standard BM25 algorithm, and needs zero extra services —
just the Python standard library's `sqlite3` module.
[`sqlite-vec`](https://github.com/asg017/sqlite-vec) does the equivalent for
vector search: it's a loadable SQLite extension that stores embeddings in a
special virtual table and answers "k nearest neighbours" queries with plain
SQL. Together, both indexes live in a single `.db` file with no separate
database server to run, deploy, or containerize — which keeps the whole
project reproducible with nothing more than a Python environment.

**Why `responses.parse` with Pydantic instead of manually parsing JSON?**
Early versions of this project asked the model to "return JSON" as free text
and parsed it by stripping markdown fences — fragile, and it broke whenever
the model's formatting drifted. OpenAI's `responses.parse` API accepts a
Pydantic model as the expected output shape and guarantees the response
matches it, which is used throughout the evaluation code (ground-truth
question generation, LLM-as-judge scoring).

## Project structure

```
llm_project/
├── crawler/              # crawl website → clean text → chunk into sections
│   ├── crawler.py
│   └── chunker.py
├── indexes/               # build & query the keyword + vector indexes
│   ├── db.py               # shared SQLite connection, base chunks table
│   ├── load_chunks.py
│   ├── keyword_index.py    # FTS5 (BM25)
│   └── build_vector_index.py  # sqlite-vec + OpenAI embeddings
├── rag/                    # retrieve → prompt → generate
│   ├── retriever.py         # keyword / vector / hybrid (RRF)
│   ├── query_rewriter.py     # bonus: LLM query rewriting
│   ├── reranker.py            # bonus: LLM re-ranking
│   ├── prompts.py               # strict / flexible prompt variants
│   ├── llm.py                     # chat completion + structured output wrapper
│   └── pipeline.py                 # answer(question) -> {answer, sources}
├── eval/                    # retrieval & LLM evaluation
│   ├── evaluation_utils.py    # parallel map, structured LLM calls, cost tracking
│   ├── generate_ground_truth.py
│   ├── retrieval_eval.py
│   ├── llm_eval.py
│   └── query_enhancement_eval.py  # rewrite/rerank ablation
├── app/                      # interface
│   ├── streamlit_app.py
│   └── pages/1_Dashboard.py
├── monitoring/
│   ├── feedback_store.py
│   └── dashboard_data.py
├── data/
│   ├── raw_html/, cleaned/    # crawler output
│   ├── chunks/                 # chunked output (chunks.jsonl)
│   ├── eval/                    # ground truth + eval results
│   └── index/                    # SQLite DB (gitignored — rebuild locally)
├── pyproject.toml / uv.lock
├── Dockerfile / docker-compose.yml / .dockerignore
└── .env.example
```

## Setup

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/getting-started/installation/),
an OpenAI API key.

```bash
git clone <this-repo>
cd llm_project
uv sync
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

`uv sync` installs every dependency at the exact pinned versions in `uv.lock`,
so the environment is reproducible across machines.

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Embeddings, generation, and LLM-as-judge evaluation |

## Running the pipeline

The whole ingestion pipeline — crawl → chunk → build keyword index → build
vector index — runs as a single automated entrypoint:

```bash
uv run python run_ingestion.py
```

This takes a few minutes (the crawler is intentionally polite, with a 1s delay
between requests) and makes real, small-cost calls to the OpenAI embeddings
API in its last step. Each stage is also a plain function
(`crawl_site`, `chunk_all`, `build_keyword_index`, `build_vector_index`)
importable from [`run_ingestion.py`](run_ingestion.py) if you want to run or
re-run a single stage instead — e.g. after editing the chunker, you only need
`chunk_all()` and the two index-build steps, not a full re-crawl.

Then launch the app:

```bash
uv run streamlit run app/streamlit_app.py
```

### Running with Docker instead

No local Python/uv setup needed — just Docker and a `.env` file (see
[Setup](#setup) above for creating one).

```bash
# build and start the app (serves on http://localhost:8501)
docker compose up app

# run the ingestion pipeline in a container instead of locally
docker compose --profile ingestion run ingestion
```

The `data/` folder is mounted as a volume into both containers, so anything
the ingestion container writes (the SQLite knowledge base, crawled pages) is
immediately visible to the app container, and persists on your machine after
the containers stop.

> **Windows troubleshooting:** if `docker compose build` fails with
> `ERROR: invalid file request ...`, this is a known Docker Desktop /
> BuildKit file-sync bug on Windows, unrelated to this project. Work around
> it by disabling BuildKit for the build:
> `DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker compose build`

## Example usage

**Q: How much do international students pay for undergraduate study?**

> The exact tuition fees for international undergraduate students at the
> University of Auckland are provided on the university's webpage dedicated to
> international student fees. You can find detailed information about prices
> for undergraduate study for international students at this URL: ...
>
> Source: https://www.auckland.ac.nz/en/study/fees-and-money-matters/tuition-fees/international-student-fees.html

**Q: When is the deadline for fees payment?**

> The exact deadlines for paying your fees at the University of Auckland
> depend on the semester or quarter of study you are enrolled in... check the
> "Paying your fees" page... [links to the two relevant pages]

*(Screenshots of the chat UI and dashboard: add your own here —
`![chat UI](docs/screenshot-chat.png)` — after running the app locally.)*

## Evaluation

### Retrieval evaluation

Three retrieval strategies were compared against a **2,667-question ground
truth set**, generated by asking an LLM to write realistic student questions
for each of the 889 chunks (3 per chunk), then checking whether each method's
top-5 results contained the chunk the question was generated from.

| method | hit rate | MRR |
|---|---|---|
| keyword (FTS5/BM25) | 0.789 | 0.637 |
| vector (sqlite-vec) | 0.628 | 0.510 |
| **hybrid (RRF)** | **0.813** | **0.646** |

**Hybrid wins on both metrics**, confirming the RRF fusion approach. Note that
keyword search alone outperforms vector search alone here — likely because
the ground-truth questions were generated from the chunk text itself, so they
share vocabulary with the source (favouring lexical matching). Real student
questions, phrased less like the source text, would likely favour vector
search more — a limitation of synthetic ground truth worth flagging.

*(`hit rate` = was a relevant chunk anywhere in the top 5 results? `MRR` =
mean reciprocal rank of the first relevant result — rewards ranking it higher.)*

### LLM evaluation

Two system prompt variants (`strict` vs `flexible`, see [Generation](#3-generation))
were compared on a random sample of 100 questions, with retrieval held fixed
at `hybrid`. Each answer was scored by an LLM judge on **relevance** (does it
address the question?) and **faithfulness** (is it fully supported by the
retrieved context, without adding outside facts?).

| variant | relevant | faithful | not faithful |
|---|---|---|---|
| strict | 97% | 74% | 4% |
| **flexible** | 97% | **76%** | **0%** |

Both variants are equally relevant. Counter to the initial hypothesis,
`flexible` (which permits general background knowledge) did **not** hurt
faithfulness — it had zero outright unfaithful answers in the sample, versus
4% for `strict`. `flexible` is used as the default as a result.

Both evaluation scripts track and report actual OpenAI API cost (ground truth
generation: ~$0.15; LLM eval judging: ~$0.08).

### Query rewriting & re-ranking (bonus)

Four retrieval configurations were compared on a random sample of 100
questions ([`eval/query_enhancement_eval.py`](eval/query_enhancement_eval.py)):

| configuration | hit rate | MRR |
|---|---|---|
| hybrid | 0.880 | 0.696 |
| hybrid + rewrite | 0.840 | 0.629 |
| **hybrid + rerank** | **0.900** | **0.737** |
| hybrid + rewrite + rerank | 0.880 | 0.730 |

**Re-ranking clearly helps** — the best single addition on both metrics.
**Query rewriting measurably hurts** in this evaluation, which is
counter-intuitive but explainable: the ground-truth questions were generated
*from* the chunk text (see [Retrieval evaluation](#retrieval-evaluation)
above), so they already share vocabulary with the source passage — rewriting
shifts the query away from that fortunate overlap. This doesn't necessarily
mean rewriting has no value for real users typing genuinely vague queries
(e.g. it turns *"how much for international students"* into a much more
specific search query, which looks like a clear improvement on inspection) —
but this synthetic evaluation set can't measure that benefit, since
LLM-generated ground-truth questions are already reasonably well-formed. The
default was set to follow the evidence rather than intuition:
**re-ranking on, query rewriting off**.

## Monitoring

The dashboard (second page of the Streamlit app) shows:

1. Feedback ratio (👍 vs 👎, pie chart)
2. Feedback volume over time
3. Feedback broken down by prompt variant
4. A live table of recent feedback
5. Retrieval evaluation comparison (hit rate & MRR by method)
6. LLM evaluation comparison (relevance & faithfulness by prompt variant)

Charts 5-6 are populated immediately from the evaluation runs above; charts
1-4 populate as real users rate answers in the app.

## Evaluation criteria checklist

Self-assessment against the [course rubric](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/project.md):

| Criterion | Status |
|---|---|
| Problem description | This README |
| Retrieval flow (knowledge base + LLM) | ✅ SQLite FTS5 + sqlite-vec → OpenAI |
| Retrieval evaluation (multiple approaches compared) | ✅ keyword / vector / hybrid, 2,667 questions |
| LLM evaluation (multiple prompts compared) | ✅ strict / flexible, LLM-as-judge, 100 questions |
| Interface | ✅ Full Streamlit UI |
| Ingestion pipeline | ✅ Single automated entrypoint (`run_ingestion.py`) |
| Monitoring | ✅ Feedback collection + 6-chart dashboard |
| Containerization | ✅ Full docker-compose (app + on-demand ingestion service) |
| Reproducibility | ✅ `uv.lock` pinned deps, `.env.example`, documented setup above |
| Hybrid search (bonus) | ✅ RRF |
| Re-ranking (bonus) | ✅ LLM re-ranker, evaluated, on by default |
| Query rewriting (bonus) | ✅ Implemented & evaluated (off by default — see evaluation) |
| Cloud deployment (bonus) | ❌ Not done |

## Known limitations / what's left

- **No cloud deployment** (bonus item).
- **Ground-truth questions are synthetic** (LLM-generated from the source
  chunks), not real student queries — see the retrieval evaluation note above.
- The `data/index/` SQLite DB is gitignored (contains OpenAI-generated
  embeddings) — running the pipeline from scratch requires an OpenAI API key
  and will make paid API calls (embeddings are cheap; the full evaluation
  suite costs well under $1 in total, see [Evaluation](#evaluation)).
- **`MAX_PAGES = 100` in `crawler.py` means real coverage gaps.** For example,
  asking "I need to book a study room in the general library" retrieves the
  Student Hubs and Libraries pages (the closest content we do have) and
  answers honestly without hallucinating a booking procedure — but the actual
  page for this (`.../student-hubs/book-a-group-study-room.html`) is linked
  from a page we crawled and matches the URL allow-list, yet was never visited
  because the breadth-first crawl hit its page cap first. This is arguably the
  correct failure mode (no hallucination, honest hedging pointing to the
  closest real pages) but reflects a genuine data-coverage gap rather than a
  retrieval or generation bug — raising `MAX_PAGES` (or crawling specific
  sections more deeply) would close gaps like this one.
