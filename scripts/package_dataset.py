import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

VERIFY_SCRIPT = REPO_ROOT / "tests" / "verify_dataset.py"


def _fmt_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def _safe_get(d: object, *keys: str, default: str = "unknown") -> str:
    cur: object = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    if cur is None:
        return default
    return str(cur)


def _resolve_metadata_path() -> Path:
    """
    Preferred metadata path (per packaging spec) is: data/dataset_metadata.json
    Current repo canonical path is: data/metadata/dataset_metadata.json

    We accept either, preferring the spec path when both exist.
    """
    legacy = DATA_DIR / "dataset_metadata.json"
    canonical = DATA_DIR / "metadata" / "dataset_metadata.json"
    if legacy.exists():
        return legacy
    if canonical.exists():
        return canonical
    # Default to spec path for error message
    return legacy


def main() -> None:
    print("====================================")
    print("DATASET PACKAGING (HF-READY EXPORT)")
    print("=======================")

    pages_sanitized = PROCESSED_DIR / "pages_sanitized.jsonl"
    graph = PROCESSED_DIR / "page_graph.gpickle"
    embeddings = EMBEDDINGS_DIR / "page_embeddings.pt"
    metadata = _resolve_metadata_path()

    REQUIRED_FILES = {
        "pages": pages_sanitized,
        "graph": graph,
        "embeddings": embeddings,
        "metadata": metadata,
    }

    # 1) Discover required dataset artifacts
    missing = [name for name, p in REQUIRED_FILES.items() if not p.exists()]
    if missing:
        print("ERROR: Missing required dataset artifacts:")
        for name in missing:
            print(f"- {name}: {REQUIRED_FILES[name]}")
        print("\nAborting export.")
        sys.exit(1)

    print("\nDiscovered artifacts:")
    for name, p in REQUIRED_FILES.items():
        size = _fmt_mb(p.stat().st_size)
        rel = p.relative_to(REPO_ROOT)
        print(f"✓ {name:10s} ({size}) — {rel}")

    # 2) Load dataset_metadata.json
    meta_obj = json.loads(metadata.read_text(encoding="utf-8"))

    # 3) Determine dataset version/name
    num_pages = _safe_get(meta_obj, "crawl", "num_pages", default="unknown")
    embedding_model = _safe_get(meta_obj, "embeddings", "model", default="unknown")
    similarity_threshold = _safe_get(meta_obj, "graph", "similarity_threshold", default="unknown")
    crawl_limit = _safe_get(meta_obj, "crawl", "crawl_limit", default="unknown")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # 4) Create export directory
    EXPORT_ROOT = REPO_ROOT / "exports"
    EXPORT_ROOT.mkdir(exist_ok=True)

    # Determine next version by scanning existing exports like: v<number>_<num_pages>_pages
    max_version = 0
    for p in EXPORT_ROOT.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith("v"):
            continue
        # Parse leading integer after "v"
        i = 1
        while i < len(name) and name[i].isdigit():
            i += 1
        if i == 1:
            continue
        try:
            ver = int(name[1:i])
        except Exception:
            continue
        if ver > max_version:
            max_version = ver

    next_version = max_version + 1
    dataset_name = f"v{next_version}_{num_pages}_pages"

    export_dir = EXPORT_ROOT / dataset_name
    if export_dir.exists():
        export_dir = EXPORT_ROOT / f"{dataset_name}-{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=False)

    # 5) Copy required artifacts
    print(f"\nCopying artifacts into: {export_dir}")
    copies = [
        (pages_sanitized, export_dir / "pages_sanitized.jsonl"),
        (graph, export_dir / "page_graph.gpickle"),
        (embeddings, export_dir / "page_embeddings.pt"),
        (metadata, export_dir / "dataset_metadata.json"),
    ]
    for src, dst in copies:
        shutil.copy2(src, dst)
        print(f"✓ {dst.name:22s} ({_fmt_mb(dst.stat().st_size)})")

    # 6) Run verify_dataset.py (do not rebuild pipeline stages)
    print("\nRunning dataset verification:")
    cmd = [sys.executable, str(VERIFY_SCRIPT)]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )

    # 7) Save verification report
    report_path = export_dir / "verification_report.txt"
    combined = ""
    if proc.stdout:
        combined += proc.stdout
    if proc.stderr:
        if combined and not combined.endswith("\n"):
            combined += "\n"
        combined += "\n[stderr]\n" + proc.stderr
    report_path.write_text(combined, encoding="utf-8")

    if proc.returncode != 0:
        print("WARNING: verify_dataset.py reported NOT READY (export will still be created).")
        print(f"Verification exit code: {proc.returncode}")
    else:
        print("✓ verification passed")
    print(f"Saved verification report: {report_path}")

    # Final instructions
    print("\n====================================")
    print("DATASET EXPORT COMPLETE")
    print("=======================")
    print(f"\nExport directory: {export_dir.resolve()}")
    print("\nReady for upload to Hugging Face.\n")
    print("Suggested workflow:")
    print("1. Create dataset repo on Hugging Face")
    print("2. Upload contents of this folder")
    print("3. Tag dataset version using page count")


if __name__ == "__main__":
    main()

