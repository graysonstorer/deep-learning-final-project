"""
Build a directed NetworkX graph from crawler artifacts.

This module is read-only relative to the dataset: it only loads exported page/link
records and constructs a NetworkX DiGraph for downstream analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple

import networkx as nx

# Configurable default paths (as requested)
PAGES_PATH = Path("../data-loading/output/pages.json")
LINKS_PATH = Path("../data-loading/output/links.json")


def _iter_json_records(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Yield dict records from a JSON or JSONL file.

    Supported formats:
    - JSON array:    [ {...}, {...}, ... ]
    - JSON object:   {"records": [ ... ]} (common wrapper)
    - JSON Lines:    one JSON object per line
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return iter(())  # empty

    # Prefer JSONL when extension hints it.
    if suffix == ".jsonl":
        def gen() -> Iterator[Dict[str, Any]]:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
        return gen()

    # Otherwise parse as JSON.
    obj = json.loads(text)
    if isinstance(obj, list):
        def gen_list() -> Iterator[Dict[str, Any]]:
            for item in obj:
                if isinstance(item, dict):
                    yield item
        return gen_list()
    if isinstance(obj, dict):
        records = obj.get("records")
        if isinstance(records, list):
            def gen_records() -> Iterator[Dict[str, Any]]:
                for item in records:
                    if isinstance(item, dict):
                        yield item
            return gen_records()
        # Single dict record (edge-case)
        def gen_one() -> Iterator[Dict[str, Any]]:
            yield obj
        return gen_one()

    return iter(())


def _default_fallback(path: Path, fallback: Path) -> Path:
    """Return path if it exists, else fallback if it exists, else path."""
    return path if path.exists() else (fallback if fallback.exists() else path)


def build_graph(
    pages_path: Path = PAGES_PATH,
    links_path: Path = LINKS_PATH,
) -> Tuple[nx.DiGraph, Dict[int, str]]:
    """
    Construct a directed graph from pages + links artifacts.

    Nodes:
      - node id: page_id (int)
      - node attributes: {"title": str}

    Edges:
      - directed (source_id -> target_id)

    Safeguards:
      - skips edges where source or target id is missing from pages

    Returns:
      (G, id_to_title)
    """
    # Support current repo's JSONL artifacts as a fallback (without changing defaults).
    # Prefer sanitized pages if present.
    pages_path = _default_fallback(pages_path, Path("data/pages_sanitized.jsonl"))
    pages_path = _default_fallback(pages_path, Path("data/pages.jsonl"))
    links_path = _default_fallback(links_path, Path("data/links.jsonl"))

    id_to_title: Dict[int, str] = {}
    title_to_id: Dict[str, int] = {}

    # Load pages
    for rec in _iter_json_records(pages_path):
        page_id = rec.get("page_id")
        title = rec.get("title")
        try:
            pid = int(page_id)
        except (TypeError, ValueError):
            continue
        if not isinstance(title, str) or not title.strip():
            continue
        t = title.strip()
        id_to_title[pid] = t
        title_to_id[t] = pid

    G = nx.DiGraph()
    for pid, t in id_to_title.items():
        G.add_node(pid, title=t)

    # Load links
    skipped_edges = 0
    added_edges = 0
    for rec in _iter_json_records(links_path):
        source_id = rec.get("source_id")
        target_id = rec.get("target_id")
        try:
            sid = int(source_id)
            tid = int(target_id)
        except (TypeError, ValueError):
            skipped_edges += 1
            continue

        if sid not in id_to_title or tid not in id_to_title:
            skipped_edges += 1
            continue

        G.add_edge(sid, tid)
        added_edges += 1

    if skipped_edges:
        print(f"[build_graph] Skipped edges (missing nodes/bad ids): {skipped_edges}")
    print(f"[build_graph] Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

    return G, id_to_title

