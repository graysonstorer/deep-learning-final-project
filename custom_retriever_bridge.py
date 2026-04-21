from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parent
CUSTOM_LOADER_ROOT = REPO_ROOT / "custom-dataset-loader"
CUSTOM_DATA_DIR = CUSTOM_LOADER_ROOT / "data"
CUSTOM_PROCESSED_DIR = CUSTOM_DATA_DIR / "processed"
CUSTOM_EMBEDDINGS_DIR = CUSTOM_DATA_DIR / "embeddings"

CUSTOM_PAGES_PATH = CUSTOM_PROCESSED_DIR / "pages_sanitized.jsonl"
CUSTOM_LINKS_PATH = CUSTOM_PROCESSED_DIR / "links_table.jsonl"
CUSTOM_GRAPH_PATH = CUSTOM_PROCESSED_DIR / "page_graph.gpickle"
CUSTOM_PAGE_EMBEDDINGS_PATH = CUSTOM_EMBEDDINGS_DIR / "page_embeddings.pt"


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def build_page_text(page: Dict[str, Any]) -> str:
    title = page.get("title") if isinstance(page.get("title"), str) else ""
    extract = ""
    if isinstance(page.get("extract_clean"), str) and page["extract_clean"].strip():
        extract = page["extract_clean"].strip()
    elif isinstance(page.get("extract"), str) and page["extract"].strip():
        extract = page["extract"].strip()

    categories: List[str] = []
    cats = page.get("categories_clean")
    if isinstance(cats, list):
        categories = [c for c in cats if isinstance(c, str) and c.strip()]
    elif isinstance(page.get("categories"), list):
        categories = [c for c in page["categories"] if isinstance(c, str) and c.strip()]

    parts: List[str] = []
    if title.strip():
        parts.append(title.strip())
    if extract:
        parts.append(extract)
    if categories:
        parts.append("Categories: " + ", ".join(categories))
    return "\n\n".join(parts).strip()


def load_custom_pages(max_pages: Optional[int] = None, pages_path: Path = CUSTOM_PAGES_PATH) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    for page in _iter_jsonl(pages_path):
        pages.append(page)
        if max_pages is not None and len(pages) >= max_pages:
            break
    return pages


def load_custom_page_texts(
    max_pages: Optional[int] = None,
    pages_path: Path = CUSTOM_PAGES_PATH,
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for page in load_custom_pages(max_pages=max_pages, pages_path=pages_path):
        try:
            page_id = int(page.get("page_id"))
        except Exception:
            continue
        text = build_page_text(page)
        if not text:
            continue
        title = page.get("title") if isinstance(page.get("title"), str) else str(page_id)
        out[page_id] = {"title": title, "text": text, "page": page}
    return out


def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def load_custom_chunk_corpus(
    max_pages: Optional[int] = None,
    chunk_size: int = 300,
    pages_path: Path = CUSTOM_PAGES_PATH,
) -> List[Tuple[str, str]]:
    corpus: List[Tuple[str, str]] = []
    page_texts = load_custom_page_texts(max_pages=max_pages, pages_path=pages_path)
    for page_id, rec in page_texts.items():
        for i, chunk in enumerate(chunk_text(rec["text"], chunk_size=chunk_size)):
            corpus.append((f"{page_id}__{i}", chunk))
    return corpus


def load_custom_embedding_artifact(path: Path = CUSTOM_PAGE_EMBEDDINGS_PATH) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(obj)}")
    return obj


def build_custom_supervision_pairs(
    max_pages: Optional[int] = None,
    max_edges: Optional[int] = None,
    pages_path: Path = CUSTOM_PAGES_PATH,
    links_path: Path = CUSTOM_LINKS_PATH,
) -> List[Tuple[str, str, int]]:
    """
    Build weak-supervision (query_text, document_text, target_page_id) tuples from
    the custom corpus without recrawling.

    Supervision sources:
    - hyperlink anchor text -> target page text
    - page title -> page text (self-pair)
    """
    page_texts = load_custom_page_texts(max_pages=max_pages, pages_path=pages_path)
    pairs: List[Tuple[str, str, int]] = []
    seen: set[Tuple[str, int]] = set()

    # Title self-pairs help preserve direct lexical grounding.
    for page_id, rec in page_texts.items():
        title = rec["title"].strip()
        if len(title) >= 3:
            key = (title.lower(), page_id)
            if key not in seen:
                seen.add(key)
                pairs.append((title, rec["text"], page_id))

    edges_seen = 0
    for row in _iter_jsonl(links_path):
        try:
            target_page_id = int(row.get("target_page_id"))
        except Exception:
            continue
        if target_page_id not in page_texts:
            continue

        anchor = row.get("anchor_clean")
        if not isinstance(anchor, str) or len(anchor.strip()) < 3:
            continue
        query = anchor.strip()
        key = (query.lower(), target_page_id)
        if key in seen:
            continue

        seen.add(key)
        pairs.append((query, page_texts[target_page_id]["text"], target_page_id))
        edges_seen += 1
        if max_edges is not None and edges_seen >= max_edges:
            break

    return pairs

