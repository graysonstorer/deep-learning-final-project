PROJECT OUTLINE — WIKIPEDIA NAVIGATION DATASET PROTOTYPE

Project Name:
Wikipedia Graph Navigation — Prototype Dataset Builder

Objective:
Build a small-scale (≈500 pages) normalized Wikipedia hyperlink graph dataset using the MediaWiki API. This dataset will support downstream experiments in semantic navigation, link prediction, and deep learning–based pathfinding.

Dataset Design Principles:

* Use a normalized schema (pages separate from links).
* Store only minimal page metadata initially.
* Restrict to article namespace (ns=0).
* Build via controlled BFS expansion from curated seed pages.
* Cap total dataset size at ~500 pages.

Directory Structure:

/wiki_graph_project
/data
pages.jsonl
links.jsonl
/seeds
seed_pages.json
/src
dataset_loader.py
/docs
dataset_spec.md
project_outline.md

Dataset Units:

Page Record (pages.jsonl):
{
"page_id": int,
"title": str
}

Link Record (links.jsonl):
{
"source_id": int,
"target_id": int
}

Acquisition Strategy:

Phase 1 — Seed ingestion
Load curated seed list.

Phase 2 — Controlled BFS crawl
Expand outward through outgoing links.

Constraints:

* Max pages: 500
* Namespace filter: articles only
* Deduplicate pages
* Deduplicate edges

Phase 3 — Storage
Write incremental results to JSONL.

Future Extensions (not in prototype):

* Page summaries
* Full text
* Embeddings
* Incoming links
* Edge weights
* Anchor text

Primary Use Cases:

* Shortest-path navigation
* Embedding-based traversal
* Neural link ranking
* Reinforcement learning agents

Implementation Stack:
Python
requests
jsonlines
tqdm (optional progress)

End of Outline
