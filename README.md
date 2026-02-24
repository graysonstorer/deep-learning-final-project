# CS6540 Deep Learning Final Project

Using transformer models to perform graph-based retrieval augmented generation (GraphRAG) tasks on data from Wikipedia. 

### Dataset acquisition (Wikipedia graph prototype)

Goal: build a small normalized Wikipedia hyperlink graph (≈500 pages) via controlled BFS expansion using the MediaWiki API (articles only, outgoing links only).

Pipeline:
- `data-loading/dataset_loader.py` → `data/pages_raw.jsonl`
- `data-loading/migrate_pages.py` → `data/pages_sanitized.jsonl`
- `data-loading/build_link_layer.py` → `data/links_table.jsonl` + `data/link_layer_report.json`

Seed setup:
- Edit `data-loading/seeds/basic_seeds.json` as a JSON array of Wikipedia article titles, for example:

```json
["Artificial intelligence", "Machine learning", "Graph theory"]
```

Run (from repo root):

```bash
python3 data-loading/dataset_loader.py
```

This will create/overwrite:
- `data/pages_raw.jsonl`

Post-crawl sanitization (Phase 1):
- Reads: `data/pages_raw.jsonl`
- Writes: `data/pages_sanitized.jsonl` (sanitized text + category cleanup + `outgoing_links` objects)

```bash
python3 data-loading/migrate_pages.py
```

Build links table:
- Reads: `data/pages_sanitized.jsonl`
- Writes: `data/links_table.jsonl`, `data/link_layer_report.json`

```bash
python3 data-loading/build_link_layer.py
```


### References: 

https://arxiv.org/pdf/2602.02053#cite.edge2025localglobalgraphrag

Dataset: [![Dataset](https://img.shields.io/badge/🤗%20Dataset-WildGraphBench-yellow)](https://huggingface.co/datasets/YOUR_HF_LINK_HERE)