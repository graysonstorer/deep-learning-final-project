# CS6540 Deep Learning Final Project

Using transformer models to perform graph-based retrieval augmented generation (GraphRAG) tasks on data from Wikipedia. 

### Dataset acquisition (Wikipedia graph prototype)

Goal: build a small normalized Wikipedia hyperlink graph (≈500 pages) via controlled BFS expansion using the MediaWiki API (articles only, outgoing links only).

New files:
- `data-loading/dataset_loader.py`: crawls Wikipedia and writes normalized JSONL outputs.
- `data-loading/seeds/seed_pages.json`: seed article titles to start the BFS crawl.
- `data/pages.jsonl`: page records (`{"page_id": int, "title": str}`).
- `data/links.jsonl`: directed link records (`{"source_id": int, "target_id": int}`).

Seed setup:
- Edit `data-loading/seeds/seed_pages.json` as a JSON array of Wikipedia article titles, for example:

```json
["Artificial intelligence", "Machine learning", "Graph theory"]
```

Run (from repo root):

```bash
python3 data-loading/dataset_loader.py
```

This will create/overwrite:
- `data/pages.jsonl`
- `data/links.jsonl`


### References: 

https://arxiv.org/pdf/2602.02053#cite.edge2025localglobalgraphrag

Dataset: [![Dataset](https://img.shields.io/badge/🤗%20Dataset-WildGraphBench-yellow)](https://huggingface.co/datasets/YOUR_HF_LINK_HERE)