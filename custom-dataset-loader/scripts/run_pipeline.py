from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def determine_version(pages: int, version: Optional[str]) -> str:
    if version and version.strip():
        return version.strip()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_pages{pages}"


def _header(title: str) -> None:
    print("\n" + "=" * 30)
    print(title)
    print("=" * 30)


def run_stage(name: str, commands: Sequence[Sequence[str]], dry_run: bool = False) -> float:
    _header(f"[PIPELINE] {name}")
    t0 = time.time()

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = env.get("PYTHONHASHSEED") or "0"

    for cmd in commands:
        cmd_list = list(cmd)
        if dry_run:
            print(f"[DRY RUN] Would execute:\n  {' '.join(cmd_list)}")
            continue
        subprocess.run(cmd_list, cwd=str(REPO_ROOT), check=True, env=env)

    dt = time.time() - t0
    if dry_run:
        print("\n[DRY RUN] Stage not executed.")
    else:
        print(f"\n✓ completed in {dt:.2f}s")
    return dt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full reproducible dataset construction pipeline.")
    p.add_argument("--pages", type=int, required=True, help="Number of Wikipedia pages to crawl.")
    p.add_argument(
        "--max_links_per_page",
        type=int,
        default=50,
        help="Maximum outgoing links retained per crawled page.",
    )
    p.add_argument("--version", type=str, default=None, help="Dataset version name (default: auto timestamp).")

    p.add_argument("--skip-crawl", action="store_true")
    p.add_argument("--skip-embeddings", action="store_true")
    p.add_argument("--skip-graph", action="store_true")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--skip-package", action="store_true")

    p.add_argument("--dry-run", action="store_true", help="Print stages without executing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pages = int(args.pages)
    if pages <= 0:
        raise SystemExit("--pages must be a positive integer")
    if int(args.max_links_per_page) <= 0:
        raise SystemExit("--max_links_per_page must be a positive integer")

    version = determine_version(pages=pages, version=args.version)

    start = time.time()
    total_runtime = 0.0
    current_stage = ""

    # Reproducibility: propagate crawl density to downstream stages via environment.
    os.environ["PIPELINE_MAX_LINKS_PER_PAGE"] = str(int(args.max_links_per_page))

    try:
        if not args.skip_crawl:
            print("[PIPELINE] Crawl Parameters")
            print(f"Pages Requested        : {pages}")
            print(f"Max Links Per Page     : {int(args.max_links_per_page)}")

            current_stage = "Stage 1: Data Loading"
            total_runtime += run_stage(
                current_stage,
                commands=[
                    [
                        sys.executable,
                        "data_loading/dataset_loader.py",
                        "--max_pages",
                        str(args.pages),
                        "--max_links_per_page",
                        str(int(args.max_links_per_page)),
                    ],
                    [sys.executable, "data_loading/migrate_pages.py"],
                    [sys.executable, "data_loading/build_link_layer.py"],
                ],
                dry_run=args.dry_run,
            )

        if not args.skip_embeddings:
            current_stage = "Stage 2: Embedding Generation"
            total_runtime += run_stage(
                current_stage,
                commands=[
                    [sys.executable, "embeddings/generate_embeddings.py", "--max_pages", str(pages)],
                ],
                dry_run=args.dry_run,
            )

        if not args.skip_graph:
            current_stage = "Stage 3: Graph Construction"
            total_runtime += run_stage(
                current_stage,
                commands=[
                    [sys.executable, "graph/build_page_graph.py"],
                ],
                dry_run=args.dry_run,
            )

        if not args.skip_verify:
            current_stage = "Stage 4: Dataset Verification"
            total_runtime += run_stage(
                current_stage,
                commands=[
                    [sys.executable, "tests/verify_dataset.py"],
                ],
                dry_run=args.dry_run,
            )

        if not args.skip_package:
            current_stage = "Stage 5: Dataset Packaging"
            total_runtime += run_stage(
                current_stage,
                commands=[
                    [sys.executable, "scripts/package_dataset.py", "--version", version],
                ],
                dry_run=args.dry_run,
            )

    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Stage failed: {current_stage or 'unknown'}")
        raise SystemExit(e.returncode)

    end = time.time()
    if args.dry_run:
        total_runtime = end - start

    print("\n" + "=" * 36)
    print("PIPELINE COMPLETE")
    print("=" * 17)
    print(f"\nDataset Version: {version}")
    print(f"Pages Requested: {pages}")
    print(f"Max Links/Page : {int(args.max_links_per_page)}")
    print(f"Export Location: exports/{version}/")
    print(f"\nTotal Runtime: {total_runtime:.2f} seconds")


if __name__ == "__main__":
    main()

