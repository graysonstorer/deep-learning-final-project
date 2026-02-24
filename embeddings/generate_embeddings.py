"""
Patch 2 — Generate text embeddings for Wikipedia graph pages.

This script loads the sanitized pages dataset (`data/pages_sanitized.jsonl`),
builds a per-page text payload (title + extract + categories), encodes it with a
configurable SentenceTransformer model, and saves embeddings to a PyTorch file.

Outputs:
  - page_embeddings.pt (torch.save)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch


def _project_root() -> Path:
    # Works locally and in Colab regardless of cwd.
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # pragma: no cover (not expected in normal python execution)
        return Path.cwd()


def load_pages(pages_path: Path, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load up to max_pages records from a JSONL pages file."""
    pages: List[Dict[str, Any]] = []
    with pages_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict):
                pages.append(rec)
            if max_pages is not None and len(pages) >= max_pages:
                break
    return pages


def _page_text_payload(page: Dict[str, Any]) -> Tuple[int, str, int]:
    """
    Build the embedding text payload for one page.

    Returns:
      (page_id, text, text_word_len)
    """
    pid_raw = page.get("page_id")
    page_id = int(pid_raw)  # will raise if invalid; caller controls input dataset

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
    else:
        cats = page.get("categories")
        if isinstance(cats, list):
            categories = [c for c in cats if isinstance(c, str) and c.strip()]

    parts: List[str] = []
    if title.strip():
        parts.append(title.strip())
    if extract:
        parts.append(extract)
    if categories:
        parts.append("Categories: " + ", ".join(categories))

    text = "\n\n".join(parts).strip()
    word_len = len(text.split()) if text else 0
    return page_id, text, word_len


def generate_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: str,
    progress_every: int,
) -> torch.Tensor:
    """
    Encode texts into embeddings using sentence-transformers.

    Returns:
      embeddings tensor of shape (N, D) on CPU.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'sentence-transformers'. Install with:\n"
            "  pip install sentence-transformers\n\n"
            "In Google Colab:\n"
            "  !pip install sentence-transformers\n"
        ) from e

    model = SentenceTransformer(model_name, device=device)

    all_embs: List[torch.Tensor] = []
    n = len(texts)
    if n == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    num_batches = math.ceil(n / batch_size)
    for b in range(num_batches):
        start = b * batch_size
        end = min(n, (b + 1) * batch_size)
        batch = texts[start:end]

        embs = model.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=False,
        )

        # Ensure CPU tensor for serialization consistency.
        all_embs.append(embs.detach().cpu())

        if progress_every > 0 and end % progress_every == 0:
            print(f"[embeddings] processed {end}/{n}")

    return torch.cat(all_embs, dim=0)


def save_embeddings(
    out_path: Path,
    page_ids: List[int],
    embeddings: torch.Tensor,
    model_name: str,
    avg_text_length_words: float,
) -> None:
    """Save embeddings artifact via torch.save."""
    payload = {
        "page_ids": page_ids,
        "embeddings": embeddings,
        "model_name": model_name,
        "avg_text_length_words": avg_text_length_words,
    }
    torch.save(payload, out_path)


def main() -> None:
    root = _project_root()
    default_pages = root / "data" / "processed" / "pages_sanitized.jsonl"
    data_dir = root / "data"
    embed_dir = data_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)
    default_out = embed_dir / "page_embeddings.pt"

    parser = argparse.ArgumentParser(description="Generate page embeddings from pages_sanitized.jsonl")
    parser.add_argument("--pages_path", type=str, default=str(default_pages), help="Path to pages_sanitized.jsonl")
    parser.add_argument("--out_path", type=str, default=str(default_out), help="Output .pt path")
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--max_pages", type=int, default=None, help="Optional max pages to process")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--progress_every", type=int, default=200, help="Log every N pages")
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device for embedding model",
    )

    args = parser.parse_args()

    pages_path = Path(args.pages_path).expanduser()
    out_path = Path(args.out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not pages_path.exists():
        raise FileNotFoundError(f"Pages file not found: {pages_path}")

    pages = load_pages(pages_path, max_pages=args.max_pages)
    print(f"[load] pages loaded: {len(pages)}")

    page_ids: List[int] = []
    texts: List[str] = []
    lengths: List[int] = []

    for rec in pages:
        try:
            pid, text, word_len = _page_text_payload(rec)
        except Exception:
            continue
        page_ids.append(pid)
        texts.append(text)
        lengths.append(word_len)

    avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0
    print(f"[stats] pages to embed: {len(texts)}")
    print(f"[stats] avg text length (words): {avg_len:.1f}")

    embeddings = generate_embeddings(
        texts=texts,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        progress_every=args.progress_every,
    )
    print(f"[embeddings] tensor shape: {tuple(embeddings.shape)}")

    save_embeddings(
        out_path=out_path,
        page_ids=page_ids,
        embeddings=embeddings,
        model_name=args.model_name,
        avg_text_length_words=avg_len,
    )
    print(f"[save] wrote: {out_path}")


if __name__ == "__main__":
    main()

