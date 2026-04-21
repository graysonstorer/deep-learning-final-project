# CS6540 Deep Learning Final Project - Custom Dataset Builder

## Directory Structure

- **`data_loading/`**: Wikipedia crawling, migration, and sanitization logic. Produces structured page records suitable for downstream processing.
- **`embeddings/`**: Generates semantic vector representations for each Wikipedia page using textual and categorical information.
- **`graph/`**: Constructs the final semantic page graph used for downstream machine learning tasks.
- **`graph/diagnostics/`**: Optional analysis and validation utilities for inspecting graph connectivity and quality.
- **`data/`**: Stores generated datasets including sanitized pages, embeddings, and graph artifacts.
- **`tests/`**: Validation scripts confirming correctness of embeddings and graph outputs.

This repository intentionally separates: **data ingestion → representation → graph construction → validation**.

## Dataset Construction Pipeline

The objective is to construct a **training-ready knowledge graph dataset** prior to model training. The Wikipedia crawler builds a small normalized graph (≈500 pages) via controlled BFS expansion using the MediaWiki API (articles only, outgoing links only).

Execution order:

1. Crawl and sanitize Wikipedia pages
   - `data_loading/dataset_loader.py` → `data/raw/pages_raw.jsonl`
   - `data_loading/migrate_pages.py` → `data/processed/pages_sanitized.jsonl`
   - `data_loading/build_link_layer.py` → `data/processed/links_table.jsonl` + `data/processed/link_layer_report.json`

2. Generate semantic page embeddings
   - `embeddings/generate_embeddings.py` → `data/embeddings/page_embeddings.pt`

3. Construct semantic similarity graph
   - `graph/build_page_graph.py` → `data/processed/page_graph.gpickle`

Seed setup:
- Edit `data_loading/seeds/basic_seeds.json` as a JSON array of Wikipedia article titles, for example:

```json
["Artificial intelligence", "Machine learning", "Graph theory"]
```

### Full pipeline runner (recommended)

This repository provides a single orchestration command that runs the full pipeline end-to-end **without reimplementing stage logic**. It calls the existing scripts in order, stops on failure, regenerates metadata, verifies training readiness, and then creates a Hugging Face–ready export bundle.

Run (from repo root):

```bash
python3 scripts/run_pipeline.py --pages 500 --max_links_per_page 50 --version v1_500_pages
```

This will produce:
- **Canonical artifacts** under `data/` (raw pages, sanitized pages, embeddings, graph, metadata)
- **An export bundle** under `exports/v1_500_pages/` (ready for upload)

Useful flags:
- `--dry-run`: print commands without executing
- `--max_links_per_page`: cap outgoing links stored per page (controls crawl graph density; default: 50)
- `--skip-crawl`, `--skip-embeddings`, `--skip-graph`, `--skip-verify`, `--skip-package`: resume partial runs
- If `--version` is omitted, an automatic version is generated like `YYYYMMDD_HHMMSS_pagesXXXX`

### Manual execution (advanced)

If you want to run stages individually (debugging or experimentation), execute the same scripts the runner orchestrates:

```bash
python3 data_loading/dataset_loader.py --max_pages 500 --max_links_per_page 50
python3 data_loading/migrate_pages.py
python3 data_loading/build_link_layer.py
python3 embeddings/generate_embeddings.py --max_pages 500
python3 graph/build_page_graph.py
python3 tests/verify_dataset.py
python3 scripts/package_dataset.py --version v1_500_pages
```

Optional diagnostics (connectivity/quality checks):

```bash
python3 -m graph.diagnostics.run_diagnostics
```

## Semantic Page Embeddings

Each Wikipedia page is transformed into a single dense vector representation. A unified document is constructed from:

- page title
- cleaned summary extract
- cleaned category labels

These components are concatenated and encoded into one embedding vector using `sentence-transformers/all-MiniLM-L6-v2`.

Rationale:

- Produces sentence-level semantic embeddings rather than isolated word vectors.
- Captures contextual meaning across entire documents.
- Computationally efficient (384-dimensional vectors).
- Widely adopted baseline model in NLP and retrieval systems.
- Suitable for similarity search, clustering, and graph construction.
- Runs efficiently on local machines or Google Colab.

Output:

- Each `page_id` maps to a normalized 384-dimensional embedding stored in `data/embeddings/page_embeddings.pt`.
- Embeddings enable quantitative semantic comparison between pages via cosine similarity.

## Semantic Page Graph

The graph represents semantic relationships between Wikipedia pages:

- Nodes correspond to individual pages.
- Edges connect pages with high embedding similarity.
- Edge weights are derived from cosine similarity between embeddings.

Graph connectivity is based on semantic similarity rather than raw hyperlink structure, which matters because it:

- captures latent conceptual relationships,
- enables graph neural network training,
- supports node classification and link prediction tasks,
- provides structure suitable for modern graph-based deep learning models.

Canonical builder: `graph/build_page_graph.py`  
Diagnostics tools: `graph/diagnostics/`

## Design Philosophy

This repository focuses on dataset engineering for graph-based machine learning. The pipeline intentionally separates:

- data collection
- feature construction
- semantic representation
- relational structure generation

Model training is decoupled so that multiple learning approaches can reuse the same dataset.

## Downstream Retriever Training

The artifact `data/embeddings/page_embeddings.pt` is a **precomputed embedding matrix** for the custom corpus. It is useful as a baseline retrieval index, but it is **not** the same thing as a fine-tuned SentenceTransformer checkpoint like the WildGraphBench retriever stored in `../all_embeddings/`.

If you want to train a new retriever from this custom corpus without recrawling:

1. Run the custom dataset pipeline here to produce:
   - `data/processed/pages_sanitized.jsonl`
   - `data/processed/links_table.jsonl`
   - `data/embeddings/page_embeddings.pt`
2. From the parent repository root, train a new retriever checkpoint:

```bash
python3 train_custom_retriever.py \
  --pages_path custom-dataset-loader/data/processed/pages_sanitized.jsonl \
  --links_path custom-dataset-loader/data/processed/links_table.jsonl \
  --output_path custom_all_embeddings
```

This creates a new SentenceTransformer checkpoint directory (for example `custom_all_embeddings/`) using weak supervision from:
- page title -> page text pairs
- hyperlink anchor text -> linked target page text pairs

To compare that new retriever against the existing WildGraphBench-tuned retriever on the WildGraphBench QA benchmark:

```bash
python3 compare_retrievers.py \
  --custom_embedder_path ./custom_all_embeddings \
  --wildgraph_embedder_path ./all_embeddings \
  --domain culture \
  --topic "Marvel Cinematic Universe"
```

Optional fixed-generator evaluation using an existing LoRA checkpoint:

```bash
python3 compare_retrievers.py \
  --custom_embedder_path ./custom_all_embeddings \
  --wildgraph_embedder_path ./all_embeddings \
  --domain culture \
  --topic "Marvel Cinematic Universe" \
  --lora_checkpoint ./tinyllama-lora-mcu
```

This comparison step lets you evaluate the custom retriever first, before deciding whether LoRA retraining is worth the cost.
