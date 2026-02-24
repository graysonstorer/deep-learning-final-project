# CS6540 Deep Learning Final Project

Using transformer models to perform graph-based retrieval augmented generation (GraphRAG) tasks on data from Wikipedia. 

### Dataset acquisition (Wikipedia graph prototype)

Goal: build a small normalized Wikipedia hyperlink graph (≈500 pages) via controlled BFS expansion using the MediaWiki API (articles only, outgoing links only).

Pipeline:
- `data_loading/dataset_loader.py` → `data/raw/pages_raw.jsonl`
- `data_loading/migrate_pages.py` → `data/processed/pages_sanitized.jsonl`
- `data_loading/build_link_layer.py` → `data/processed/links_table.jsonl` + `data/processed/link_layer_report.json`
- `embeddings/generate_embeddings.py` → `data/embeddings/page_embeddings.pt`

Seed setup:
- Edit `data_loading/seeds/basic_seeds.json` as a JSON array of Wikipedia article titles, for example:

```json
["Artificial intelligence", "Machine learning", "Graph theory"]
```

Run (from repo root):

```bash
python3 data_loading/dataset_loader.py
```

This will create/overwrite:
- `data/raw/pages_raw.jsonl`

Post-crawl sanitization (Phase 1):
- Reads: `data/raw/pages_raw.jsonl`
- Writes: `data/processed/pages_sanitized.jsonl` (sanitized text + category cleanup + `outgoing_links` objects)

```bash
python3 data_loading/migrate_pages.py
```

Build links table:
- Reads: `data/processed/pages_sanitized.jsonl`
- Writes: `data/processed/links_table.jsonl`, `data/processed/link_layer_report.json`

```bash
python3 data_loading/build_link_layer.py
```

Generate embeddings:
- Reads: `data/processed/pages_sanitized.jsonl`
- Writes: `data/embeddings/page_embeddings.pt`

```bash
python3 embeddings/generate_embeddings.py
```

## Text Embedding Strategy

This project treats each Wikipedia page as a **concept node** in a hyperlink graph, so we require a **single fixed-length vector** per page that captures *document-level semantic meaning*.

- **Motivation**: word-level embeddings (e.g., Word2Vec/GloVe) are not sufficient for node features because they do not represent the meaning of an entire passage directly, and simple aggregation (mean/sum pooling over word vectors) tends to lose context and compositional semantics that matter for concept similarity.

- **Model choice**: we use the Sentence Transformer model `sentence-transformers/all-MiniLM-L6-v2`.
  - It is a **transformer encoder** that produces dense embeddings for a full text span (sentence/paragraph/document excerpt).
  - It is trained with **contrastive objectives** for semantic similarity (bringing related texts closer and pushing unrelated texts apart), so **similar concepts cluster in vector space**.

- **Advantages for this project**:
  - Produces **one embedding per Wikipedia page** (natural node representation).
  - Captures **contextual semantics** from the page extract + category cues.
  - Enables **cosine similarity** comparisons between concepts (useful for retrieval and navigation scoring).
  - Works well as **node features for GNNs**, complementing hyperlink structure.
  - **Computationally efficient** (small model; feasible on laptop/Colab).
  - Common **academic baseline** for semantic representation learning.
  - **Fully reproducible** (local inference; no external embedding API dependency).

- **Role in the overall pipeline**:

Wikipedia text → Sentence Transformer embeddings → Graph construction (Wikipedia links) → Graph neural network training

Embeddings provide **semantic information**, while hyperlinks provide **structural information**.

- **Design rationale (offline embeddings)**: embeddings are generated during dataset construction (offline) rather than learned jointly with the graph model to:
  - reduce computational cost,
  - stabilize GNN training,
  - enable rapid experimentation by swapping graph models while keeping node features fixed.


### References: 

https://arxiv.org/pdf/2602.02053#cite.edge2025localglobalgraphrag

Dataset: [![Dataset](https://img.shields.io/badge/🤗%20Dataset-WildGraphBench-yellow)](https://huggingface.co/datasets/YOUR_HF_LINK_HERE)