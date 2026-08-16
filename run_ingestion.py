from crawler.chunker import chunk_all
from crawler.crawler import SEED_URLS, crawl_site
from indexes.build_vector_index import build as build_vector_index
from indexes.keyword_index import build as build_keyword_index


def print_step(step, total, label):
    print()
    print("=" * 40)
    print(f"Step {step}/{total}: {label}")
    print("=" * 40)


def run():
    print_step(1, 4, "Crawling auckland.ac.nz")
    crawl_site(SEED_URLS)

    print_step(2, 4, "Chunking pages into topical sections")
    chunk_all()

    print_step(3, 4, "Building keyword index (SQLite FTS5, local, free)")
    build_keyword_index()

    print_step(4, 4, "Building vector index (OpenAI embeddings — real API cost)")
    build_vector_index()

    print()
    print("Ingestion complete. Knowledge base ready at data/index/knowledge_base.db")


if __name__ == "__main__":
    run()
