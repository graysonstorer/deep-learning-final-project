"""
Wikipedia graph dataset loader.

This module crawls Wikipedia articles via the MediaWiki API and builds a small,
normalized hyperlink graph using controlled BFS expansion.

Outputs (JSONL):
- data/pages.jsonl  : {"page_id": int, "title": str}
- data/links.jsonl  : {"source_id": int, "target_id": int}
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

API_ENDPOINT = "https://en.wikipedia.org/w/api.php"

# Crawl controls
MAX_PAGES = 500
MAX_LINKS_PER_PAGE = 50

# Polite delay between HTTP requests (seconds)
REQUEST_DELAY_S = 0.1

# Default paths
SEEDS_PATH = Path("seeds/seed_pages.json")
PAGES_OUT_PATH = Path("data/pages.jsonl")
LINKS_OUT_PATH = Path("data/links.jsonl")

# Module-level file handles used by write_* functions (opened in crawl_dataset).
_PAGES_FH: Optional[Any] = None
_LINKS_FH: Optional[Any] = None


def load_seeds(path: str | Path) -> List[str]:
    """
    Load seed article titles from a JSON file.

    The file must contain a JSON array of strings, e.g.:
    ["Artificial intelligence", "Machine learning"]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Seed file not found at {p}. Expected a JSON list of article titles."
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) and x.strip() for x in data):
        raise ValueError(f"Seed file {p} must be a JSON array of non-empty strings.")
    # Normalize whitespace; keep canonicalization to API.
    return [x.strip() for x in data]


def _api_get(params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform a MediaWiki API GET request with basic error handling and rate limiting."""
    resp = requests.get(API_ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_S)
    return resp.json()


def process_api_response(response: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], List[str]]:
    """
    Extract (page_id, title, outgoing_link_titles) from a MediaWiki 'query' response.

    Notes:
    - Only outgoing links in article namespace (ns=0) are returned.
    - Missing pages return (None, None, []).
    """
    query = response.get("query") or {}
    pages = query.get("pages") or {}
    if not isinstance(pages, dict) or not pages:
        return None, None, []

    # The pages dict is keyed by page_id as string, or "-1" for missing pages.
    page_obj = next(iter(pages.values()))
    if not isinstance(page_obj, dict):
        return None, None, []

    if page_obj.get("missing") is not None or page_obj.get("pageid") is None:
        return None, None, []

    page_id = int(page_obj["pageid"])
    title = page_obj.get("title")
    if not isinstance(title, str) or not title.strip():
        return None, None, []

    link_titles: List[str] = []
    for link in page_obj.get("links") or []:
        if not isinstance(link, dict):
            continue
        if link.get("ns") != 0:
            continue
        t = link.get("title")
        if isinstance(t, str) and t.strip():
            link_titles.append(t)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    uniq_links: List[str] = []
    for t in link_titles:
        if t not in seen:
            seen.add(t)
            uniq_links.append(t)

    return page_id, title, uniq_links


def fetch_page_links(title: str) -> Tuple[Optional[int], Optional[str], List[str]]:
    """
    Fetch outgoing article links for a given Wikipedia page title.

    Handles MediaWiki pagination via the 'continue' field.
    Returns (page_id, canonical_title, outgoing_link_titles).
    """
    all_links: List[str] = []
    plcontinue: Optional[str] = None
    page_id: Optional[int] = None
    canonical_title: Optional[str] = None

    while True:
        params: Dict[str, Any] = {
            "action": "query",
            "prop": "links",
            "titles": title,
            "pllimit": "max",
            "plnamespace": 0,
            "format": "json",
            "redirects": 1,
        }
        if plcontinue:
            params["plcontinue"] = plcontinue

        response = _api_get(params)
        pid, ptitle, links = process_api_response(response)
        if pid is None or ptitle is None:
            # Missing page or unexpected response.
            return None, None, []
        page_id = pid
        canonical_title = ptitle
        all_links.extend(links)

        cont = response.get("continue")
        if isinstance(cont, dict) and cont.get("plcontinue"):
            plcontinue = str(cont["plcontinue"])
            continue
        break

    # Deduplicate while preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for t in all_links:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return page_id, canonical_title, uniq


def _chunked(seq: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), chunk_size):
        yield list(seq[i : i + chunk_size])


def _fetch_page_ids(titles: Sequence[str]) -> Dict[str, int]:
    """
    Resolve page IDs for a list of titles.

    Returns mapping {canonical_title: page_id} for existing pages (ns=0 is not enforced here;
    callers should only pass article titles).
    """
    if not titles:
        return {}

    # MediaWiki supports multiple titles joined with '|'
    joined = "|".join(titles)
    params: Dict[str, Any] = {
        "action": "query",
        "titles": joined,
        "format": "json",
        "redirects": 1,
    }
    response = _api_get(params)
    query = response.get("query") or {}
    pages = query.get("pages") or {}

    out: Dict[str, int] = {}
    if not isinstance(pages, dict):
        return out

    for page_obj in pages.values():
        if not isinstance(page_obj, dict):
            continue
        if page_obj.get("missing") is not None:
            continue
        pid = page_obj.get("pageid")
        t = page_obj.get("title")
        if isinstance(pid, int) and isinstance(t, str) and t.strip():
            out[t] = pid
    return out


def write_page(record: Dict[str, Any]) -> None:
    """Write a single page record as one JSON line to data/pages.jsonl."""
    global _PAGES_FH
    if _PAGES_FH is None:
        raise RuntimeError("Pages writer is not initialized. Call crawl_dataset() first.")
    _PAGES_FH.write(json.dumps(record, ensure_ascii=False) + "\n")
    _PAGES_FH.flush()


def write_link(record: Dict[str, Any]) -> None:
    """Write a single link record as one JSON line to data/links.jsonl."""
    global _LINKS_FH
    if _LINKS_FH is None:
        raise RuntimeError("Links writer is not initialized. Call crawl_dataset() first.")
    _LINKS_FH.write(json.dumps(record, ensure_ascii=False) + "\n")
    _LINKS_FH.flush()


def crawl_dataset(
    seeds_path: str | Path = SEEDS_PATH,
    pages_out_path: str | Path = PAGES_OUT_PATH,
    links_out_path: str | Path = LINKS_OUT_PATH,
    max_pages: int = MAX_PAGES,
    max_links_per_page: int = MAX_LINKS_PER_PAGE,
) -> None:
    """
    Crawl Wikipedia using a controlled BFS expansion and write JSONL outputs incrementally.

    Behavior:
    - Breadth-first expansion from seed titles
    - Caps dataset at max_pages unique pages (by page_id)
    - Extracts up to max_links_per_page outgoing links per crawled page
    - Ensures no duplicate pages and no duplicate edges in the output files
    """
    seeds = load_seeds(seeds_path)
    if not seeds:
        raise ValueError("Seed list is empty.")

    # Canonicalize seeds before queue initialization to avoid duplicate enqueue paths
    # (e.g., raw title + redirected canonical title).
    resolved_seeds = _fetch_page_ids(seeds)  # {canonical_title: page_id}
    if not resolved_seeds:
        raise ValueError("No valid seed pages could be resolved to page IDs.")
    if len(resolved_seeds) > max_pages:
        # Respect dataset cap even at initialization time.
        resolved_seeds = dict(list(resolved_seeds.items())[:max_pages])

    pages_path = Path(pages_out_path)
    links_path = Path(links_out_path)
    pages_path.parent.mkdir(parents=True, exist_ok=True)
    links_path.parent.mkdir(parents=True, exist_ok=True)

    # Known pages are those included in the dataset (written or reserved),
    # capped at max_pages. This prevents edges pointing to out-of-dataset nodes.
    known_pages: Dict[int, str] = {}  # page_id -> title
    title_to_id: Dict[str, int] = {}  # canonical_title -> page_id
    visited: Set[int] = set()  # crawled pages by ID
    enqueued_titles: Set[str] = set()
    edges_written: Set[Tuple[int, int]] = set()

    # Prepopulate dataset with resolved seed pages and initialize queue using canonical titles.
    for title, page_id in resolved_seeds.items():
        known_pages[page_id] = title
        title_to_id[title] = page_id
        enqueued_titles.add(title)

    q: Deque[str] = deque(resolved_seeds.keys())

    global _PAGES_FH, _LINKS_FH
    _PAGES_FH = pages_path.open("w", encoding="utf-8")
    _LINKS_FH = links_path.open("w", encoding="utf-8")
    try:
        # Write resolved seed pages first (one record per page).
        for page_id, title in known_pages.items():
            write_page({"page_id": page_id, "title": title})

        while q and len(visited) < max_pages:
            current_title = q.popleft()

            page_id, canonical_title, outgoing_titles = fetch_page_links(current_title)
            if page_id is None or canonical_title is None:
                # Skip missing pages.
                continue

            # Include this page in dataset if possible.
            if page_id not in known_pages:
                if len(known_pages) >= max_pages:
                    # Dataset full; do not add new nodes (and thus do not crawl further).
                    break
                known_pages[page_id] = canonical_title
                title_to_id[canonical_title] = page_id
                write_page({"page_id": page_id, "title": canonical_title})
            else:
                title_to_id[canonical_title] = page_id

            if page_id in visited:
                continue
            visited.add(page_id)

            # Restrict links per page (dedup already handled).
            outgoing_titles = outgoing_titles[:max_links_per_page]

            # Resolve IDs for outgoing titles in batches to reduce requests.
            resolved: Dict[str, int] = {}
            for chunk in _chunked(outgoing_titles, 50):
                resolved.update(_fetch_page_ids(chunk))

            # Add targets to dataset (reserve) until dataset cap is reached.
            for target_title, target_id in resolved.items():
                if target_id not in known_pages:
                    if len(known_pages) >= max_pages:
                        continue
                    known_pages[target_id] = target_title
                    title_to_id[target_title] = target_id
                    write_page({"page_id": target_id, "title": target_title})

                edge = (page_id, target_id)
                if edge in edges_written:
                    continue
                edges_written.add(edge)
                write_link({"source_id": page_id, "target_id": target_id})

                # BFS enqueue for future crawl, but only if it's in-dataset and not yet visited.
                if target_id not in visited and target_title not in enqueued_titles:
                    q.append(target_title)
                    enqueued_titles.add(target_title)

            # Progress logging
            print(
                f"Dataset pages: {len(known_pages)} | "
                f"Crawled pages: {len(visited)} | "
                f"Queue size: {len(q)} | "
                f"Edges written: {len(edges_written)}"
            )
    finally:
        if _PAGES_FH is not None:
            _PAGES_FH.close()
        if _LINKS_FH is not None:
            _LINKS_FH.close()
        _PAGES_FH = None
        _LINKS_FH = None


if __name__ == "__main__":
    crawl_dataset()

