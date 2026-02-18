# CS6540 Deep Learning Final Project

Using transformer models to perform graph-based retrieval augmented generation (GraphRAG) tasks on data from Wikipedia. 

### Dataset acquisition (Wikipedia graph prototype)

Goal: build a small normalized Wikipedia hyperlink graph (≈500 pages) via controlled BFS expansion using the MediaWiki API (articles only, outgoing links only).

New files:
- `src/dataset_loader.py`: crawls Wikipedia and writes normalized JSONL outputs.
- `seeds/seed_pages.json`: seed article titles to start the BFS crawl.
- `data/pages.jsonl`: page records (`{"page_id": int, "title": str}`).
- `data/links.jsonl`: directed link records (`{"source_id": int, "target_id": int}`).

Run:

```bash
python3 src/dataset_loader.py
```


### References: 

https://arxiv.org/pdf/2602.02053#cite.edge2025localglobalgraphrag

Dataset: [![Dataset](https://img.shields.io/badge/🤗%20Dataset-WildGraphBench-yellow)](https://huggingface.co/datasets/YOUR_HF_LINK_HERE)