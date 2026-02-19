"""
Migration script: sanitize and enrich page records post-crawl.

Reads:
  - data/pages.jsonl
  - data/links.jsonl

Writes:
  - data/pages_sanitized.jsonl

This is intentionally separate from the crawler so you can backfill new fields
without a re-crawl.
"""

from __future__ import annotations

import json
import re
import string
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Set

import requests

PAGES_IN_PATH = Path("data/pages.jsonl")
LINKS_IN_PATH = Path("data/links.jsonl")
PAGES_OUT_PATH = Path("data/pages_sanitized.jsonl")

WIKI_REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
HEADERS = {"User-Agent": "WikiGraphCrawler/0.1 (academic project)"}

LOG_EVERY = 50

CATEGORY_NOISE_SUBSTRINGS = [
    "articles",
    "wikipedia",
    "cs1",
    "short description",
    "wikidata",
    "use dmy",
    "use mdy",
    "pages using",
    "webarchive",
    "commons category",
    "indefinitely",
    "errors",
    "maint",
    "isbn",
    "template",
    "tracking",
    "citation",
]

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "in",
    "on",
    "to",
    "for",
    "with",
    "by",
    "from",
    "at",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "into",
    "over",
    "under",
    "about",
}


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            yield obj


def _rest_summary_extract(title: str) -> str:
    """Fetch plaintext summary from Wikipedia REST endpoint; return '' on failure."""
    try:
        url = WIKI_REST_SUMMARY + requests.utils.quote(title, safe="")
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return ""
        data = r.json()
        extract = data.get("extract")
        return extract.strip() if isinstance(extract, str) else ""
    except Exception:
        return ""


_CITATION_RE = re.compile(r"\[(\d+|citation needed)\]", flags=re.IGNORECASE)
_PAREN_IPA_RE = re.compile(
    r"\([^)]*(pronounced|ipa|/|ˈ|ˌ|ə|ɔ|ɪ|ʊ|ɛ|æ|ɑ|ɒ|ʌ|ŋ)[^)]*\)",
    flags=re.IGNORECASE,
)


def _clean_extract_for_training(text: str) -> str:
    s = text.lower()
    s = s.replace("\n", " ")
    s = _CITATION_RE.sub("", s)
    s = _PAREN_IPA_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _category_is_noise(cat: str) -> bool:
    c = cat.lower()
    return any(sub in c for sub in CATEGORY_NOISE_SUBSTRINGS)


def _tokenize_categories(categories_clean: List[str]) -> List[str]:
    tokens: Set[str] = set()
    trans = str.maketrans({ch: " " for ch in string.punctuation})
    for cat in categories_clean:
        s = cat.lower().translate(trans)
        for tok in s.split():
            tok = tok.strip()
            if not tok or tok in STOPWORDS:
                continue
            tokens.add(tok)

    # Optional lemmatization if nltk is available and usable.
    try:
        from nltk.stem import WordNetLemmatizer  # type: ignore

        lemmatizer = WordNetLemmatizer()
        return sorted({lemmatizer.lemmatize(t) for t in tokens})
    except Exception:
        return sorted(tokens)


def sanitize_page_record(page_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a page record into the Phase 1 enriched schema.

    Expects page_dict to contain:
      - page_id, title
      - extract (optional), categories / categories_raw (optional)
      - outgoing_links (optional)
    """
    pid_raw = page_dict.get("page_id")
    title = page_dict.get("title") if isinstance(page_dict.get("title"), str) else ""
    try:
        page_id = int(pid_raw)
    except (TypeError, ValueError):
        return {}

    extract_raw = page_dict.get("extract") if isinstance(page_dict.get("extract"), str) else ""
    extract = extract_raw.strip()

    # Extract completion
    rest_extract = ""
    if not extract:
        rest_extract = _rest_summary_extract(title)
        extract = rest_extract.strip() if rest_extract else ""

    has_extract = bool(extract)
    if not extract:
        # Surrogate text for usability; keep has_extract accurate (False)
        extract = title

    extract_clean = _clean_extract_for_training(extract)
    extract_length = len(extract_clean.split()) if has_extract else 0

    # Categories
    cats_in = page_dict.get("categories_raw")
    if not isinstance(cats_in, list):
        cats_in = page_dict.get("categories") if isinstance(page_dict.get("categories"), list) else []
    categories_raw = [c for c in cats_in if isinstance(c, str)]
    categories_clean = [c for c in categories_raw if c.strip() and not _category_is_noise(c)]
    category_tokens = _tokenize_categories(categories_clean)

    # Links
    outgoing = page_dict.get("outgoing_links")
    if not isinstance(outgoing, list):
        outgoing = []
    outgoing_links: List[int] = []
    for x in outgoing:
        try:
            outgoing_links.append(int(x))
        except (TypeError, ValueError):
            continue
    link_count = len(outgoing_links)

    is_stub = extract_length < 40

    return {
        "page_id": page_id,
        "title": title,
        "extract": extract_raw if isinstance(extract_raw, str) else "",
        "extract_clean": extract_clean,
        "extract_length": extract_length,
        "categories_raw": categories_raw,
        "categories_clean": categories_clean,
        "category_tokens": category_tokens,
        "outgoing_links": outgoing_links,
        "link_count": link_count,
        "has_extract": has_extract,
        "is_stub": is_stub,
    }


def main() -> None:
    if not PAGES_IN_PATH.exists():
        raise FileNotFoundError(f"Missing input pages file: {PAGES_IN_PATH}")
    if not LINKS_IN_PATH.exists():
        raise FileNotFoundError(f"Missing input links file: {LINKS_IN_PATH}")

    # Build outgoing link lists per source_id (no dedup here by design).
    outgoing_map: DefaultDict[int, List[int]] = defaultdict(list)
    for rec in _iter_jsonl(LINKS_IN_PATH):
        try:
            sid = int(rec.get("source_id"))
            tid = int(rec.get("target_id"))
        except (TypeError, ValueError):
            continue
        outgoing_map[sid].append(tid)

    total = 0
    has_extract_count = 0
    total_extract_len = 0
    total_clean_cats = 0
    total_links = 0

    PAGES_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PAGES_OUT_PATH.open("w", encoding="utf-8") as fh:
        for rec in _iter_jsonl(PAGES_IN_PATH):
            pid = rec.get("page_id")
            try:
                page_id = int(pid)
            except (TypeError, ValueError):
                continue

            rec = dict(rec)
            rec["outgoing_links"] = outgoing_map.get(page_id, [])

            enriched = sanitize_page_record(rec)
            if not enriched:
                continue

            fh.write(json.dumps(enriched, ensure_ascii=False) + "\n")

            total += 1
            if enriched["has_extract"]:
                has_extract_count += 1
            total_extract_len += int(enriched["extract_length"])
            total_clean_cats += len(enriched["categories_clean"])
            total_links += int(enriched["link_count"])

            if total % LOG_EVERY == 0:
                pct = (has_extract_count / total) * 100.0 if total else 0.0
                avg_len = (total_extract_len / total) if total else 0.0
                avg_cats = (total_clean_cats / total) if total else 0.0
                avg_links = (total_links / total) if total else 0.0
                print(
                    f"[sanitize] {total} pages | % with extracts: {pct:.1f} | "
                    f"avg extract length: {avg_len:.1f} | avg categories_clean: {avg_cats:.1f} | "
                    f"avg link_count: {avg_links:.1f}"
                )

    pct = (has_extract_count / total) * 100.0 if total else 0.0
    print("\n=== Sanitization Complete ===")
    print("Pages:", total)
    print(f"% pages with extracts: {pct:.1f}")
    print("Output:", str(PAGES_OUT_PATH))


if __name__ == "__main__":
    main()

