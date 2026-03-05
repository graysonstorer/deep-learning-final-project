from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_METADATA_PATH = Path("data") / "metadata" / "dataset_metadata.json"


def build_dataset_metadata(
    *,
    num_pages: int,
    embedding_model_name: Optional[str],
    embedding_dimension: Optional[int],
    similarity_metric: Optional[str],
    similarity_threshold: Optional[float],
    graph_type: Optional[str],
    num_nodes: Optional[int],
    num_edges: Optional[int],
    crawl_limit: Optional[int],
    avg_text_length_words: Optional[float] = None,
    additional_notes: Optional[str] = None,
) -> Dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()

    def _int_or_none(x: Any) -> Optional[int]:
        if x is None:
            return None
        try:
            return int(x)
        except Exception as e:
            raise TypeError(f"Expected int-like value, got {x!r}") from e

    def _float_or_none(x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
            v = float(x)
        except Exception as e:
            raise TypeError(f"Expected float-like value, got {x!r}") from e
        if not (v is None or (v == v and abs(v) != float("inf"))):
            raise ValueError(f"Non-finite float value: {x!r}")
        return v

    metadata: Dict[str, Any] = {
        "dataset_info": {
            "created_at": created_at,
            "dataset_version": "auto",
        },
        "crawl": {
            "num_pages": _int_or_none(num_pages),
            "crawl_limit": _int_or_none(crawl_limit),
        },
        "embeddings": {
            "model": embedding_model_name,
            "dimension": _int_or_none(embedding_dimension),
            "normalization": "L2",
        },
        "graph": {
            "type": graph_type,
            "similarity_metric": similarity_metric,
            "similarity_threshold": _float_or_none(similarity_threshold),
            "num_nodes": _int_or_none(num_nodes),
            "num_edges": _int_or_none(num_edges),
        },
        "statistics": {
            "avg_text_length_words": _float_or_none(avg_text_length_words),
        },
    }

    if additional_notes is not None:
        metadata["dataset_info"]["additional_notes"] = str(additional_notes)

    return metadata


def save_dataset_metadata(metadata: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
    """
    Save dataset metadata JSON alongside artifacts.

    The canonical output is:
      data/metadata/dataset_metadata.json
    """
    out = output_path or DEFAULT_METADATA_PATH
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metadata, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out


def load_dataset_metadata(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    p = Path(path or DEFAULT_METADATA_PATH)
    if not p.exists():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"Metadata at {p} is not a JSON object")
    return obj

