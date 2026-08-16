import json
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
import time
import requests
from bs4 import BeautifulSoup

##configuration



HEADERS = {
    "User-Agent": "UoA Student Navigator (Educational Project)"
}

SEED_URLS = [
    "https://www.auckland.ac.nz/en/study.html",
    "https://www.auckland.ac.nz/en/students.html",
    "https://www.auckland.ac.nz/en/on-campus.html",
    "https://www.auckland.ac.nz/en/news.html"
]

# raised from 100: fixing the fragment-stripping bug below surfaces the
# programme-finder directory (~200 individual programme pages) that was
# previously invisible to the crawler entirely, so the old budget would have
# starved coverage of everything else
MAX_PAGES = 300

ALLOWED_PATHS = [
    "/en/study/",
    "/en/students/",
    "/en/on-campus/",
    "/en/news.html"
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_HTML_DIR = PROJECT_ROOT / "data" / "raw_html"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"

RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR.mkdir(parents=True, exist_ok=True)


#filter links to avoid the ones like tel: mail to etc.
def should_visit(url: str) -> bool:

    if url.startswith("tel:"):
        return False

    if url.startswith("mailto:"):
        return False

    if url.endswith(".pdf"):
        return False

    if not url.startswith("https://www.auckland.ac.nz"):
        return False

    return any(path in url for path in ALLOWED_PATHS)


# save files
def save_raw_html(url: str, html: str):
    parsed = urlparse(url)

    filename = (
        parsed.path
        .strip("/")
        .replace("/", "_")
        .replace(".html", "")
    )
    with open(RAW_HTML_DIR / f"{filename}.html", "w", encoding="utf-8") as f:
        f.write(html)


def save_document(document):

    parsed = urlparse(document["url"])

    filename = (
        parsed.path
        .strip("/")
        .replace("/", "_")
        .replace(".html", "")
    )

    with open(CLEANED_DIR / f"{filename}.json", "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)




def crawl_page(url: str):

    print(f"Crawling: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)

    except requests.RequestException as e:
        print(f"Request failed {e}")
        return None    

    if response.status_code != 200:
        print(f"Failed ({response.status_code})")
        return None

    html = response.text

    save_raw_html(url, html)

    soup = BeautifulSoup(html, "lxml")

    main = soup.find("main", id="main")

    if main is None:
        print("No <main id='main'> found.")
        return None

    # Remove unnecessary tags

    for tag in main(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.get_text(strip=True)

    text = main.get_text(separator="\n", strip=True)

    headings = []

    for heading in main.find_all(["h2", "h3"]):

        heading_text = heading.get_text(strip=True)

        if heading_text:
            headings.append(heading_text)

    # Remove duplicate headings

    headings = list(dict.fromkeys(headings))

    links = []

    seen_links = set()

    for a in main.find_all("a", href=True):

        # strip the fragment rather than rejecting the whole URL for having
        # one — same-page anchors collapse to the page itself (already
        # filtered by the visited-set check), while URLs that use a fragment
        # for client-side state (e.g. a filtered results view) keep the
        # meaningful query string that actually determines what's returned
        href, _fragment = urldefrag(urljoin(url, a["href"]))

        if href in seen_links:
            continue

        seen_links.add(href)

        links.append({
            "text": a.get_text(strip=True),
            "href": href
        })

    return {
        "url": url,
        "title": title,
        "text": text,
        "headings": headings,
        "links": links
    }

def crawl_site(seed_urls):

    queue = list(seed_urls)

    visited = set()

    documents = []

    while queue and len(documents) < MAX_PAGES:

        current_url = queue.pop(0)

        if current_url in visited:
            continue

        document = crawl_page(current_url)

        visited.add(current_url)

        if document is None:
            continue

        save_document(document)

        documents.append(document)

        for link in document["links"]:

            href = link["href"]

            if (
                should_visit(href)
                and href not in visited
                and href not in queue
            ):
                queue.append(href)

        time.sleep(1)            

    print()

    print("=" * 40)

    print(f"Pages Crawled : {len(documents)}")

    print(f"Visited URLs  : {len(visited)}")

    print("=" * 40)


if __name__ == "__main__":

    crawl_site(SEED_URLS)