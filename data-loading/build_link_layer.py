"""
Post-processing: build a training-ready link layer (edges).

Reads:
  - data/pages_sanitized.jsonl

Writes:
  - data/links_table.jsonl
  - data/link_layer_report.json

This does not modify the crawler; it operates on saved artifacts only.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd


DATA_DIR = Path("data")
PAGES_SANITIZED = DATA_DIR / "pages_sanitized.jsonl"

LINKS_OUT_PATH = DATA_DIR / "links_table.jsonl"
REPORT_OUT_PATH = DATA_DIR / "link_layer_report.json"


_CITATION_RE = re.compile(r"\[\d+\]")
_WS_RE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """
    Normalize a Wikipedia title for lookup.

    Rules:
    - lowercase
    - strip whitespace
    - replace underscores with spaces
    - remove fragment identifiers (#section)
    - decode URL encoding if present
    """
    t = title.strip()
    if "#" in t:
        t = t.split("#", 1)[0]
    t = urllib.parse.unquote(t)
    t = t.replace("_", " ")
    t = _WS_RE.sub(" ", t).strip().lower()
    return t


def clean_anchor(text: str) -> str:
    """Clean anchor text for storage (nullable upstream)."""
    s = _CITATION_RE.sub("", text)
    s = _WS_RE.sub(" ", s).strip()
    return s or ""


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            yield obj


def _choose_pages_path() -> Path:
    return PAGES_SANITIZED


def _parse_outgoing_link_item(item: Any) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
    """
    Parse one outgoing_links item.

    Returns:
      (target_title, anchor_clean, section_clean, target_page_id)
    where either target_title or target_page_id is present.
    """
    if isinstance(item, int):
        return None, None, None, item
    if isinstance(item, str):
        return item, None, None, None
    if isinstance(item, dict):
        # Support sanitized schema (target_title + anchor_clean + section_clean).
        target_title = None
        for k in ("target_title", "title", "target", "targetTitle"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                target_title = v.strip()
                break

        anchor_clean = None
        for k in ("anchor_clean", "anchor_text", "anchor", "text"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                anchor_clean = v.strip()
                break

        section_clean = None
        for k in ("section_clean", "section"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                section_clean = v.strip()
                break

        target_page_id = None
        for k in ("target_page_id", "target_id", "targetPageId", "targetPageID"):
            v = item.get(k)
            try:
                if v is not None:
                    target_page_id = int(v)
                    break
            except (TypeError, ValueError):
                continue

        return target_title, anchor_clean, section_clean, target_page_id

    return None, None, None, None


def build_link_layer(
    pages_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Build links table (JSONL) and validation report.
    Returns a report dict (also written to REPORT_OUT_PATH).
    """
    pages_path = pages_path or _choose_pages_path()
    if not pages_path.exists():
        raise FileNotFoundError(f"Pages dataset not found: {pages_path}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: title -> page_id lookup map
    title_to_id: Dict[str, int] = {}
    page_ids: set[int] = set()
    for rec in _iter_jsonl(pages_path):
        try:
            pid = int(rec.get("page_id"))
        except (TypeError, ValueError):
            continue
        title = rec.get("title")
        if isinstance(title, str) and title.strip():
            title_to_id[normalize_title(title)] = pid
        page_ids.add(pid)

    # Step 2/3: parse outgoing_links into deduped edges
    edge_to_features: Dict[Tuple[int, int], Dict[str, str]] = {}
    skipped_missing_target = 0
    skipped_self_loops = 0
    skipped_unparseable = 0
    emitted_edges_raw = 0

    sum_reported_link_count = 0
    pages_seen = 0

    for rec in _iter_jsonl(pages_path):
        pages_seen += 1
        try:
            source_id = int(rec.get("page_id"))
        except (TypeError, ValueError):
            continue

        outgoing = rec.get("outgoing_links")
        if not isinstance(outgoing, list):
            outgoing = []

        # Use the stored link_count if present; otherwise fall back to list length.
        try:
            sum_reported_link_count += int(rec.get("link_count", len(outgoing)))
        except (TypeError, ValueError):
            sum_reported_link_count += len(outgoing)

        for item in outgoing:
            target_title, anchor_clean, section_clean, target_page_id = _parse_outgoing_link_item(item)

            if target_page_id is None:
                if target_title is None:
                    skipped_unparseable += 1
                    continue
                target_page_id = title_to_id.get(normalize_title(target_title))
                if target_page_id is None:
                    skipped_missing_target += 1
                    continue

            emitted_edges_raw += 1

            if source_id == target_page_id:
                skipped_self_loops += 1
                continue

            if target_page_id not in page_ids:
                skipped_missing_target += 1
                continue

            # Ensure anchors are non-null for training readiness.
            ac = ""
            if isinstance(anchor_clean, str) and anchor_clean.strip():
                ac = clean_anchor(anchor_clean)
            else:
                ac = clean_anchor(target_title or "")
            sc = ""
            if isinstance(section_clean, str) and section_clean.strip():
                sc = clean_anchor(section_clean)

            edge = (source_id, target_page_id)
            if edge not in edge_to_features:
                edge_to_features[edge] = {"anchor_clean": ac, "section_clean": sc}
            else:
                # Prefer non-empty fields if we didn't have them.
                if not edge_to_features[edge].get("anchor_clean") and ac:
                    edge_to_features[edge]["anchor_clean"] = ac
                if not edge_to_features[edge].get("section_clean") and sc:
                    edge_to_features[edge]["section_clean"] = sc

    # Build links dataframe
    rows = [
        {
            "source_page_id": s,
            "target_page_id": t,
            "anchor_clean": feats.get("anchor_clean", ""),
            "section_clean": feats.get("section_clean", ""),
        }
        for (s, t), feats in edge_to_features.items()
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["target_page_id", "source_page_id"], kind="mergesort")
    # Persist as JSONL (one record per line).
    with LINKS_OUT_PATH.open("w", encoding="utf-8") as fh:
        for row in df.to_dict(orient="records"):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Step 4: integrity validation
    total_links_rows = int(len(df))

    orphan_edges = 0  # should be zero by construction
    self_loops = 0  # should be zero by construction
    if not df.empty:
        orphan_edges = int(
            (~df["source_page_id"].isin(page_ids) | ~df["target_page_id"].isin(page_ids)).sum()
        )
        self_loops = int((df["source_page_id"] == df["target_page_id"]).sum())

    report = {
        "pages_path": str(pages_path),
        "pages_count": int(len(page_ids)),
        "edges_emitted_raw": int(emitted_edges_raw),
        "links_rows_deduped": int(total_links_rows),
        "skipped_missing_target": int(skipped_missing_target),
        "skipped_self_loops": int(skipped_self_loops),
        "skipped_unparseable": int(skipped_unparseable),
        "sum_reported_link_count": int(sum_reported_link_count),
        "orphan_edges_detected": int(orphan_edges),
        "self_loops_detected": int(self_loops),
        "outputs": {
            "links_table_jsonl": str(LINKS_OUT_PATH),
            "report_json": str(REPORT_OUT_PATH),
        },
    }

    REPORT_OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Hard integrity checks
    if orphan_edges != 0:
        raise RuntimeError(f"Orphan edges detected after build: {orphan_edges}")
    if self_loops != 0:
        raise RuntimeError(f"Self loops detected after build: {self_loops}")

    return report


def main() -> None:
    report = build_link_layer()
    print("=== Link Layer Build Complete ===")
    print("Pages:", report["pages_count"])
    print("Links (deduped):", report["links_rows_deduped"])
    print("Skipped missing targets:", report["skipped_missing_target"])
    print("Skipped self-loops:", report["skipped_self_loops"])
    print("Wrote:", report["outputs"]["links_table_jsonl"])
    print("Report:", report["outputs"]["report_json"])


if __name__ == "__main__":
    main()

