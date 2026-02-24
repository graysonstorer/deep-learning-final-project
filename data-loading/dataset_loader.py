"""
Wikipedia graph dataset loader.

This module crawls Wikipedia articles via the MediaWiki API and builds a small,
normalized hyperlink graph using controlled BFS expansion.

Outputs (JSONL):
- data/pages_raw.jsonl : canonical raw page snapshots with semantic links
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

API_ENDPOINT = "https://en.wikipedia.org/w/api.php"

# Wikimedia API policy: identify your client with a clear User-Agent.
HEADERS = {
    "User-Agent": "WikiGraphCrawler/0.1 (szimpfer@uvm.edu)",
}

# Crawl controls
MAX_PAGES = 500
MAX_LINKS_PER_PAGE = 50

# Randomization controls
RANDOMIZE_LINK_SAMPLING = True
RANDOM_SEED = 42  # Set to None for full stochasticity

# Polite delay between HTTP requests (seconds)
REQUEST_DELAY_S = 0.1

# Default paths (resolve relative to this file so execution from repo root works)
DATA_LOADING_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_LOADING_DIR.parent

SEEDS_PATH = DATA_LOADING_DIR / "seeds" / "basic_seeds.json"
PAGES_OUT_PATH = REPO_ROOT / "data" / "pages_raw.jsonl"

# Module-level file handles used by write_* functions (opened in crawl_dataset).
_PAGES_FH: Optional[Any] = None


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
    resp = requests.get(API_ENDPOINT, params=params, headers=HEADERS, timeout=30)
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


def _is_excluded_title(title: str) -> bool:
    """Return True if page title should be excluded from dataset."""
    t = title.strip().lower()
    if not t:
        return True

    # List pages
    if t.startswith("list of"):
        return True

    # Glossaries / indices
    if t.startswith("glossary of"):
        return True

    if "index of" in t:
        return True

    # Disambiguation handled separately via pageprops, but keep defensive fallback.
    if "(disambiguation)" in t:
        return True

    return False


def _fetch_page_ids(
    titles: Sequence[str],
    stats: Optional[Dict[str, int]] = None,
    debug: bool = False,
) -> Dict[str, int]:
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
        "prop": "pageprops",
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
        if isinstance(page_obj.get("pageprops"), dict) and "disambiguation" in page_obj["pageprops"]:
            if stats is not None:
                stats["excluded_disambig_count"] = stats.get("excluded_disambig_count", 0) + 1
            continue
        pid_raw = page_obj.get("pageid")
        t = page_obj.get("title")

        # Normalize page_id type (MediaWiki may return str or int)
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            continue

        if isinstance(t, str) and t.strip():
            if _is_excluded_title(t):
                if stats is not None:
                    stats["excluded_title_count"] = stats.get("excluded_title_count", 0) + 1
                continue

            out[t] = pid
            if debug:
                print(f"Resolved seed: {t} → {pid}")
    return out


_HEADING_RE = re.compile(r"^(?P<eq>={2,6})\s*(?P<title>[^=].*?)\s*(?P=eq)\s*$")
_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")
_IGNORED_PREFIXES = (
    "File:",
    "Category:",
    "Template:",
    "Help:",
    "Wikipedia:",
    "Special:",
)


def _parse_wikitext_sections_and_links(wikitext: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parse wikitext to extract:
    - sections: [{"section_title": str, "level": int}, ...]
    - links: [{"target_title": str, "anchor": str, "section": str | None}, ...]
    """
    sections: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    seen_targets: Set[str] = set()

    current_section: Optional[str] = None

    for line in wikitext.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            level = len(m.group("eq"))
            title = m.group("title").strip()
            if title:
                current_section = title
                sections.append({"section_title": title, "level": level})
            continue

        for lm in _WIKILINK_RE.finditer(line):
            inner = lm.group(1).strip()
            if not inner:
                continue

            # Ignore interwiki prefixes and leading colon forms.
            if inner.startswith(":"):
                inner = inner[1:].lstrip()

            # Split on first pipe: [[target|anchor]]
            if "|" in inner:
                target_part, anchor_part = inner.split("|", 1)
                anchor = anchor_part.strip()
            else:
                target_part, anchor = inner, ""

            target_part = target_part.strip()
            if not target_part:
                continue

            # Drop fragment identifiers
            if "#" in target_part:
                target_part = target_part.split("#", 1)[0].strip()
            if not target_part:
                continue

            # Ignore namespaces/pseudo-pages
            if any(target_part.startswith(p) for p in _IGNORED_PREFIXES):
                continue
            if ":" in target_part:
                # Defensive: ignore any remaining namespace-style links.
                continue

            target_title = target_part.replace("_", " ").strip()
            if not target_title:
                continue

            if target_title in seen_targets:
                continue
            seen_targets.add(target_title)

            if not anchor:
                anchor = target_title

            links.append(
                {
                    "target_title": target_title,
                    "anchor": anchor,
                    "section": current_section,
                }
            )

    return sections, links


def fetch_page_raw(title: str) -> Tuple[Optional[int], Optional[str], Optional[str], List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fetch canonical page metadata + wikitext and parse sections + semantic links.

    Returns:
      (page_id, canonical_title, extract_or_none, categories, sections, links)
    """
    clcontinue: Optional[str] = None
    extract: Optional[str] = None
    categories: List[str] = []
    page_id: Optional[int] = None
    canonical_title: Optional[str] = None
    wikitext: str = ""

    attempts_left = 2
    while True:
        try:
            params: Dict[str, Any] = {
                "action": "query",
                "format": "json",
                "titles": title,
                "redirects": 1,
                "prop": "extracts|categories|revisions|pageprops",
                "exintro": 1,
                "explaintext": 1,
                "cllimit": "max",
                "rvprop": "content",
                "rvslots": "main",
                "rvlimit": 1,
            }
            if clcontinue:
                params["clcontinue"] = clcontinue

            resp = _api_get(params)
        except Exception as e:
            attempts_left -= 1
            if attempts_left <= 0:
                print(f"[raw] WARNING: failed to fetch page '{title}' ({e})")
                return None, None, None, [], [], []
            time.sleep(0.5)
            continue

        pages = (resp.get("query") or {}).get("pages") or {}
        if not isinstance(pages, dict) or not pages:
            return None, None, None, [], [], []
        page_obj = next(iter(pages.values()))
        if not isinstance(page_obj, dict):
            return None, None, None, [], [], []
        if page_obj.get("missing") is not None:
            return None, None, None, [], [], []

        # Disambiguation skip
        if isinstance(page_obj.get("pageprops"), dict) and "disambiguation" in page_obj["pageprops"]:
            return None, None, None, [], [], []

        try:
            page_id = int(page_obj.get("pageid"))
        except (TypeError, ValueError):
            return None, None, None, [], [], []

        t = page_obj.get("title")
        canonical_title = t.strip() if isinstance(t, str) and t.strip() else None

        ex = page_obj.get("extract")
        if extract is None and isinstance(ex, str) and ex.strip():
            extract = ex

        # Categories may be continued across requests.
        cats = page_obj.get("categories") or []
        if isinstance(cats, list):
            for c in cats:
                if not isinstance(c, dict):
                    continue
                ct = c.get("title")
                if not isinstance(ct, str) or not ct.strip():
                    continue
                name = ct
                if name.startswith("Category:"):
                    name = name[len("Category:") :]
                name = name.strip()
                if name and name not in categories:
                    categories.append(name)

        # Wikitext present in revisions (only need once)
        if not wikitext:
            revs = page_obj.get("revisions") or []
            if isinstance(revs, list) and revs:
                rev0 = revs[0]
                if isinstance(rev0, dict):
                    slots = rev0.get("slots")
                    if isinstance(slots, dict):
                        main = slots.get("main")
                        if isinstance(main, dict):
                            wt = main.get("*") or main.get("content")
                            if isinstance(wt, str):
                                wikitext = wt
                    # Some API formats may return content at top-level
                    if not wikitext:
                        wt = rev0.get("*") or rev0.get("content")
                        if isinstance(wt, str):
                            wikitext = wt

        cont = resp.get("continue")
        if isinstance(cont, dict) and cont.get("clcontinue"):
            clcontinue = str(cont["clcontinue"])
            continue
        break

    sections, links = _parse_wikitext_sections_and_links(wikitext or "")
    return page_id, canonical_title, extract if (isinstance(extract, str) and extract.strip()) else None, categories, sections, links

def _clean_extract(text: str) -> str:
    """Minimal sanitation for extracts before storing."""
    s = text.strip().replace("\n", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def fetch_page_metadata(page_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Batch fetch page intro extracts + categories for up to ~50 pages at once.

    Returns:
      {page_id: {"extract": str, "categories": [str]}}
    """
    if not page_ids:
        return {}

    # Default structure for all requested page ids (ensures keys exist).
    out: Dict[int, Dict[str, Any]] = {
        int(pid): {"extract": "", "categories": []} for pid in page_ids if isinstance(pid, int)
    }
    if not out:
        return {}

    # Track per-page category deduplication during continuation.
    seen_cats: Dict[int, Set[str]] = {pid: set() for pid in out.keys()}

    params_base: Dict[str, Any] = {
        "action": "query",
        "format": "json",
        "pageids": "|".join(str(pid) for pid in out.keys()),
        "prop": "extracts|categories",
        "exintro": 1,
        "explaintext": 1,
        "cllimit": "max",
    }

    cont: Dict[str, Any] = {}
    attempts_left = 2  # retry once

    while True:
        try:
            params = dict(params_base)
            params.update(cont)
            response = _api_get(params)
        except Exception as e:
            attempts_left -= 1
            if attempts_left <= 0:
                print(f"[metadata] WARNING: metadata fetch failed ({e}). Using empty defaults.")
                return out
            time.sleep(0.5)
            continue

        pages = (response.get("query") or {}).get("pages") or {}
        if isinstance(pages, dict):
            for page_obj in pages.values():
                if not isinstance(page_obj, dict):
                    continue
                pid_raw = page_obj.get("pageid")
                try:
                    pid = int(pid_raw)
                except (TypeError, ValueError):
                    continue
                if pid not in out:
                    continue

                extract = page_obj.get("extract")
                if isinstance(extract, str) and extract.strip():
                    out[pid]["extract"] = _clean_extract(extract)

                cats = page_obj.get("categories") or []
                if isinstance(cats, list):
                    for c in cats:
                        if not isinstance(c, dict):
                            continue
                        ct = c.get("title")
                        if not isinstance(ct, str) or not ct.strip():
                            continue
                        name = ct
                        if name.startswith("Category:"):
                            name = name[len("Category:") :]
                        name = name.strip()
                        if not name or name in seen_cats[pid]:
                            continue
                        seen_cats[pid].add(name)
                        out[pid]["categories"].append(name)

        cont_obj = response.get("continue")
        if isinstance(cont_obj, dict) and cont_obj.get("clcontinue"):
            # MediaWiki continuation requires the 'continue' token as well.
            cont = {
                "clcontinue": cont_obj.get("clcontinue"),
                "continue": cont_obj.get("continue"),
            }
            continue

        break

    return out


def write_page(record: Dict[str, Any]) -> None:
    """Write a single page record as one JSON line to data/pages_raw.jsonl."""
    global _PAGES_FH
    if _PAGES_FH is None:
        raise RuntimeError("Pages writer is not initialized. Call crawl_dataset() first.")

    page_id = record.get("page_id")
    try:
        pid = int(page_id)
    except (TypeError, ValueError):
        return

    title = record.get("title") if isinstance(record.get("title"), str) else ""
    extract = record.get("extract")
    if not (extract is None or isinstance(extract, str)):
        extract = None

    categories_raw = record.get("categories")
    categories: List[str] = []
    if isinstance(categories_raw, list):
        categories = [c for c in categories_raw if isinstance(c, str)]

    sections_raw = record.get("sections")
    sections: List[Dict[str, Any]] = []
    if isinstance(sections_raw, list):
        for s in sections_raw:
            if not isinstance(s, dict):
                continue
            st = s.get("section_title")
            lvl = s.get("level")
            if isinstance(st, str) and st.strip():
                try:
                    ilvl = int(lvl)
                except (TypeError, ValueError):
                    ilvl = 0
                sections.append({"section_title": st.strip(), "level": ilvl})

    links_raw = record.get("links")
    links: List[Dict[str, Any]] = []
    if isinstance(links_raw, list):
        for l in links_raw:
            if not isinstance(l, dict):
                continue
            tt = l.get("target_title")
            anch = l.get("anchor")
            sec = l.get("section")
            if isinstance(tt, str) and tt.strip():
                link_obj: Dict[str, Any] = {
                    "target_title": tt.strip(),
                    "anchor": anch if isinstance(anch, str) and anch.strip() else tt.strip(),
                    "section": sec if isinstance(sec, str) and sec.strip() else None,
                }
                links.append(link_obj)

    out_record = {
        "page_id": pid,
        "title": title,
        "extract": extract if (isinstance(extract, str) and extract.strip()) else None,
        "categories": categories,
        "sections": sections,
        "links": links,
    }

    _PAGES_FH.write(json.dumps(out_record, ensure_ascii=False) + "\n")
    _PAGES_FH.flush()


def crawl_dataset(
    seeds_path: str | Path = SEEDS_PATH,
    pages_out_path: str | Path = PAGES_OUT_PATH,
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
    print("SEEDS PATH:", seeds_path)

    seeds = load_seeds(seeds_path)

    print("SEED COUNT:", len(seeds))
    print("FIRST 5:", seeds[:5])

    if not seeds:
        raise ValueError("Seed list is empty.")

    excluded_title_count = 0
    excluded_disambig_count = 0
    stats: Dict[str, int] = {}

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # Canonicalize seeds before queue initialization to avoid duplicate enqueue paths
    # (e.g., raw title + redirected canonical title).
    #
    # MediaWiki limits the number of titles per query (commonly 50 for non-bot clients),
    # so resolve seeds in batches to avoid empty results from API errors.
    resolved_seeds: Dict[str, int] = {}
    for chunk in _chunked(seeds, 50):
        resolved_seeds.update(_fetch_page_ids(chunk, stats=stats, debug=True))  # {canonical_title: page_id}

    print("RESOLVED:", len(resolved_seeds))
    print("FIRST 5:", list(resolved_seeds.items())[:5])

    if not resolved_seeds:
        raise ValueError("No valid seed pages could be resolved to page IDs.")

    # Filter titles during seed resolution (defensive; also handled in _fetch_page_ids).
    pre_filter_seed_count = len(resolved_seeds)
    resolved_seeds = {
        title: pid for title, pid in resolved_seeds.items() if not _is_excluded_title(title)
    }
    excluded_title_count += pre_filter_seed_count - len(resolved_seeds)
    if len(resolved_seeds) > max_pages:
        # Respect dataset cap even at initialization time.
        resolved_seeds = dict(list(resolved_seeds.items())[:max_pages])

    pages_path = Path(pages_out_path)
    pages_path.parent.mkdir(parents=True, exist_ok=True)

    visited: Set[int] = set()  # crawled pages by ID
    enqueued_titles: Set[str] = set(resolved_seeds.keys())
    q: Deque[str] = deque(resolved_seeds.keys())

    global _PAGES_FH
    _PAGES_FH = pages_path.open("w", encoding="utf-8")
    try:
        while q and len(visited) < max_pages:
            current_title = q.popleft()

            page_id, canonical_title, extract, categories, sections, links = fetch_page_raw(current_title)
            if page_id is None or canonical_title is None:
                continue
            if _is_excluded_title(canonical_title):
                excluded_title_count += 1
                continue
            if page_id in visited:
                continue
            visited.add(page_id)

            # Filter candidate link targets by exclusion rules before sampling.
            filtered_links: List[Dict[str, Any]] = []
            for l in links:
                tt = l.get("target_title")
                if not isinstance(tt, str) or not tt.strip():
                    continue
                if _is_excluded_title(tt):
                    excluded_title_count += 1
                    continue
                filtered_links.append(l)

            original_links_count = len(filtered_links)

            # De-alphabetize link selection before applying the per-page cap.
            sampled_links = list(filtered_links)
            if RANDOMIZE_LINK_SAMPLING:
                random.shuffle(sampled_links)
                buckets: Dict[str, List[Dict[str, Any]]] = {}
                for l in sampled_links:
                    tt = l.get("target_title")
                    if not isinstance(tt, str) or not tt:
                        continue
                    buckets.setdefault(tt[0].upper(), []).append(l)

                balanced: List[Dict[str, Any]] = []
                while buckets and len(balanced) < max_links_per_page:
                    for ch in list(buckets.keys()):
                        if buckets[ch]:
                            balanced.append(buckets[ch].pop(0))
                            if len(balanced) >= max_links_per_page:
                                break
                        if not buckets.get(ch):
                            buckets.pop(ch, None)
                sampled_links = balanced
            else:
                sampled_links = sampled_links[:max_links_per_page]

            print(
                f"[LINK SAMPLING] {canonical_title}: selected {len(sampled_links)} / "
                f"{original_links_count} links"
            )

            # Write raw page snapshot (no sanitization here).
            write_page(
                {
                    "page_id": page_id,
                    "title": canonical_title,
                    "extract": extract,
                    "categories": categories,
                    "sections": sections,
                    "links": sampled_links,
                }
            )

            # BFS enqueue next-degree expansion targets.
            for l in sampled_links:
                tt = l.get("target_title")
                if not isinstance(tt, str) or not tt.strip():
                    continue
                if tt not in enqueued_titles and len(visited) + len(q) < max_pages * 5:
                    q.append(tt)
                    enqueued_titles.add(tt)

            # Progress logging
            print(
                f"Dataset pages: {len(visited)} | "
                f"Crawled pages: {len(visited)} | "
                f"Queue size: {len(q)} | "
                f"Pages written: {len(visited)}"
            )

        excluded_title_count += stats.get("excluded_title_count", 0)
        excluded_disambig_count += stats.get("excluded_disambig_count", 0)
        print("=== Crawl Hygiene Summary ===")
        print(f"Excluded titles: {excluded_title_count}")
        print(f"Excluded disambiguation pages: {excluded_disambig_count}")
    finally:
        if _PAGES_FH is not None:
            _PAGES_FH.close()
        _PAGES_FH = None


if __name__ == "__main__":
    crawl_dataset()

