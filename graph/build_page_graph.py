"""
Patch 3 — Build a page graph from embeddings + structural links.

Constructs a directed graph where:
- Nodes are Wikipedia pages (page_id)
- Edges include structural hyperlink edges (from links_table.jsonl)
- Optionally adds semantic similarity edges based on cosine similarity between page embeddings

Inputs (expected):
- data/embeddings/page_embeddings.pt
- data/processed/pages_sanitized.jsonl
- data/processed/links_table.jsonl

Output:
- data/processed/page_graph.gpickle

Run:
  python build_page_graph.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import networkx as nx
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_metadata import build_dataset_metadata, load_dataset_metadata, save_dataset_metadata  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBED_DIR = DATA_DIR / "embeddings"

DEFAULT_EMBEDDINGS_PATH = EMBED_DIR / "page_embeddings.pt"
DEFAULT_PAGES_PATH = PROCESSED_DIR / "pages_sanitized.jsonl"
DEFAULT_LINKS_PATH = PROCESSED_DIR / "links_table.jsonl"
DEFAULT_OUT_PATH = PROCESSED_DIR / "page_graph.gpickle"


def load_embeddings(path: Path = DEFAULT_EMBEDDINGS_PATH) -> Dict[str, Any]:
    """Load Patch 2 embeddings artifact (torch.save payload)."""
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "page_ids" not in obj or "embeddings" not in obj:
        raise ValueError(f"Unexpected embeddings format in {path}")
    if not isinstance(obj["page_ids"], list) or not torch.is_tensor(obj["embeddings"]):
        raise ValueError(f"Unexpected embeddings types in {path}")
    return obj


def load_pages(path: Path = DEFAULT_PAGES_PATH) -> List[Dict[str, Any]]:
    """Load sanitized pages dataset (JSONL)."""
    pages: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict):
                pages.append(rec)
    return pages


def iter_structural_links(path: Path = DEFAULT_LINKS_PATH) -> Iterator[Tuple[int, int, Dict[str, Any]]]:
    """
    Yield structural links from links_table.jsonl as (source_id, target_id, attrs).
    Expects keys: source_page_id, target_page_id, anchor_clean, section_clean.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            try:
                s = int(rec.get("source_page_id"))
                t = int(rec.get("target_page_id"))
            except (TypeError, ValueError):
                continue
            attrs = {
                "weight": 1.0,
                "edge_type": "structural",
            }
            anch = rec.get("anchor_clean")
            sec = rec.get("section_clean")
            if isinstance(anch, str) and anch.strip():
                attrs["anchor_clean"] = anch
            if isinstance(sec, str) and sec.strip():
                attrs["section_clean"] = sec
            yield s, t, attrs


def _build_page_graph_from_loaded(
    pages: List[Dict[str, Any]],
    embeddings_obj: Optional[Dict[str, Any]],
    links_path: Path,
    top_k: int,
    add_semantic: bool,
    similarity_threshold: Optional[float],
) -> Tuple[nx.DiGraph, Dict[int, str]]:
    """
    Build directed graph with nodes from pages, structural edges from links_table,
    and optional semantic edges (top-k cosine similarity).
    """
    G = nx.DiGraph()

    # Nodes
    page_ids: List[int] = []
    id_to_title: Dict[int, str] = {}
    for p in pages:
        try:
            pid = int(p.get("page_id"))
        except (TypeError, ValueError):
            continue
        page_ids.append(pid)
        title = p.get("title") if isinstance(p.get("title"), str) else ""
        id_to_title[pid] = title
        cats = p.get("categories_clean")
        if not isinstance(cats, list):
            cats = p.get("categories", [])
        categories = [c for c in cats if isinstance(c, str)] if isinstance(cats, list) else []
        G.add_node(pid, title=title, categories=categories)

    page_id_set = set(page_ids)

    # Structural edges
    structural_edges_added = 0
    if links_path.exists():
        for s, t, attrs in iter_structural_links(links_path):
            if s in page_id_set and t in page_id_set:
                G.add_edge(s, t, **attrs)
                structural_edges_added += 1
    else:
        print(f"[warn] structural links file not found: {links_path}")

    print(f"[structural] edges added: {structural_edges_added}")

    # Semantic edges
    if not add_semantic or top_k <= 0:
        return G, id_to_title
    if embeddings_obj is None:
        print("[semantic] embeddings missing; skipping semantic edges")
        return G, id_to_title

    emb_page_ids: List[int] = [int(x) for x in embeddings_obj["page_ids"]]
    emb_tensor: torch.Tensor = embeddings_obj["embeddings"].detach().cpu()
    if emb_tensor.ndim != 2:
        raise ValueError("Embeddings tensor must be 2D (N, D)")

    emb_index = {pid: i for i, pid in enumerate(emb_page_ids)}
    usable_page_ids = [pid for pid in page_ids if pid in emb_index]
    if len(usable_page_ids) < 2:
        print("[semantic] not enough pages with embeddings to add semantic edges")
        return G, id_to_title

    X = torch.stack([emb_tensor[emb_index[pid]] for pid in usable_page_ids], dim=0)
    # Normalize for cosine similarity using dot product.
    X = torch.nn.functional.normalize(X, p=2, dim=1)
    sim = X @ X.T  # (M, M)
    sim.fill_diagonal_(-1.0)

    semantic_edges_added = 0
    for i, src_pid in enumerate(usable_page_ids):
        vals, idxs = torch.topk(sim[i], k=min(top_k, sim.shape[1] - 1))
        for v, j in zip(vals.tolist(), idxs.tolist()):
            tgt_pid = usable_page_ids[j]
            weight = float(v)
            if similarity_threshold is not None and weight < float(similarity_threshold):
                continue
            if G.has_edge(src_pid, tgt_pid):
                # Preserve structural edge and add semantic contribution to weight.
                G[src_pid][tgt_pid]["weight"] = float(G[src_pid][tgt_pid].get("weight", 0.0)) + weight
                # Keep original edge_type if present.
                if "edge_type" not in G[src_pid][tgt_pid]:
                    G[src_pid][tgt_pid]["edge_type"] = "structural"
            else:
                G.add_edge(src_pid, tgt_pid, weight=weight, edge_type="semantic")
                semantic_edges_added += 1

        if (i + 1) % 100 == 0:
            print(f"[semantic] processed {i+1}/{len(usable_page_ids)}")

    print(f"[semantic] edges added: {semantic_edges_added}")
    print(
        f"[summary] nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
        f"(structural≈{structural_edges_added}, semantic_added={semantic_edges_added})"
    )
    return G, id_to_title


def save_graph(G: nx.DiGraph, path: Path = DEFAULT_OUT_PATH) -> None:
    """Save graph to gpickle using pickle (networkx 3.x compatible)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[save] {path} (nodes={G.number_of_nodes()}, edges={G.number_of_edges()})")


def build_page_graph(
    pages_path: Optional[Path] = None,
    links_path: Optional[Path] = None,
    embeddings_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    top_k: int = 5,
    similarity_threshold: Optional[float] = None,
    add_semantic: bool = True,
    save: bool = True,
) -> Tuple[nx.DiGraph, Dict[int, str]]:
    """
    Canonical graph construction pipeline (Patch 3).

    Returns (G, id_to_title).
    """
    # If called as a script (no args provided), parse CLI args.
    if (
        pages_path is None
        and links_path is None
        and embeddings_path is None
        and out_path is None
        and len(sys.argv) > 1
    ):
        parser = argparse.ArgumentParser(description="Build page graph (structural + semantic edges).")
        parser.add_argument("--embeddings", type=str, default=str(DEFAULT_EMBEDDINGS_PATH))
        parser.add_argument("--pages", type=str, default=str(DEFAULT_PAGES_PATH))
        parser.add_argument("--links", type=str, default=str(DEFAULT_LINKS_PATH))
        parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_PATH))
        parser.add_argument("--top_k", type=int, default=top_k)
        parser.add_argument(
            "--similarity_threshold",
            type=float,
            default=similarity_threshold,
            help="Optional minimum cosine similarity to keep a semantic edge",
        )
        parser.add_argument("--no_semantic", action="store_true", help="Disable semantic similarity edges")
        args = parser.parse_args()

        pages_path = Path(args.pages)
        links_path = Path(args.links)
        embeddings_path = Path(args.embeddings)
        out_path = Path(args.out)
        top_k = int(args.top_k)
        similarity_threshold = args.similarity_threshold
        add_semantic = not args.no_semantic

    pages_path = pages_path or DEFAULT_PAGES_PATH
    links_path = links_path or DEFAULT_LINKS_PATH
    embeddings_path = embeddings_path or DEFAULT_EMBEDDINGS_PATH
    out_path = out_path or DEFAULT_OUT_PATH

    pages = load_pages(pages_path)

    embeddings_obj: Optional[Dict[str, Any]] = None
    if add_semantic and embeddings_path.exists():
        embeddings_obj = load_embeddings(embeddings_path)

    G, id_to_title = _build_page_graph_from_loaded(
        pages=pages,
        embeddings_obj=embeddings_obj,
        links_path=links_path,
        top_k=top_k,
        add_semantic=add_semantic,
        similarity_threshold=similarity_threshold,
    )

    if save:
        save_graph(G, out_path)

    # Dataset metadata (reproducibility): update graph fields without erasing prior info.
    meta_path = PROJECT_ROOT / "data" / "metadata" / "dataset_metadata.json"
    existing = load_dataset_metadata(meta_path)
    created_at = datetime.now(timezone.utc).isoformat()

    if existing is None:
        emb_model = None
        emb_dim = None
        avg_len = None
        if embeddings_obj is not None:
            emb_model = embeddings_obj.get("model_name") if isinstance(embeddings_obj.get("model_name"), str) else None
            emb_t = embeddings_obj.get("embeddings")
            if torch.is_tensor(emb_t) and emb_t.ndim == 2:
                emb_dim = int(emb_t.shape[1])
            try:
                avg_len = float(embeddings_obj.get("avg_text_length_words"))
            except Exception:
                avg_len = None

        existing = build_dataset_metadata(
            num_pages=len(pages),
            embedding_model_name=emb_model,
            embedding_dimension=emb_dim,
            similarity_metric="cosine",
            similarity_threshold=similarity_threshold,
            graph_type="semantic_similarity",
            num_nodes=G.number_of_nodes(),
            num_edges=G.number_of_edges(),
            crawl_limit=len(pages),
            avg_text_length_words=avg_len,
            additional_notes="Metadata created during graph construction (embedding stage metadata missing).",
        )
    else:
        # Update timestamp to reflect the latest rebuild.
        if isinstance(existing.get("dataset_info"), dict):
            existing["dataset_info"]["created_at"] = created_at
        else:
            existing["dataset_info"] = {"created_at": created_at, "dataset_version": "auto"}

        existing["graph"] = {
            "type": "semantic_similarity",
            "similarity_metric": "cosine",
            "similarity_threshold": similarity_threshold,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
        }

    save_dataset_metadata(existing, output_path=meta_path)
    print(f"[metadata] updated: {meta_path}")

    return G, id_to_title


if __name__ == "__main__":
    build_page_graph()

