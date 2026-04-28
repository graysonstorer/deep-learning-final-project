# CS6540 Deep Learning Final Project

Using transformer models to study graph-based retrieval-augmented generation on Wikipedia-derived data. The current project focuses on two linked goals:

1. fine-tuning pretrained retrieval embeddings for graph/RAG-style question answering, and
2. LoRA fine-tuning a small language model on retrieved context-answer pairs.

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-WildGraphBench-yellow)](https://arxiv.org/pdf/2602.02053#cite.edge2025localglobalgraphrag)

## Project Scope

The main experimental workflow currently uses `WildGraphBench` as the source benchmark and retrieval corpus. Training and analysis in this repository center on:

- fine-tuning a pretrained `sentence-transformers/all-MiniLM-L6-v2` retrieval model on `WildGraphBench`
- LoRA fine-tuning of `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- comparison of RAG performance with base vs. fine-tuned embeddings
- plotting loss curves and retrieval metrics across training runs

The repository also includes a separate dataset-construction module in `custom-dataset-loader/`. That module has its own README and documents dataset creation in detail; the main project README only covers how it connects to the training pipeline here.

## Repository Layout

- `train_embeddings.py`: fine-tunes a pretrained retrieval model on a single `WildGraphBench` domain
- `train_embeddings_all_subjects.py`: scales that fine-tuning workflow across multiple benchmark domains
- `train_custom_retriever.py`: fine-tunes a new retrieval model from the custom dataset artifacts produced by `custom-dataset-loader/`
- `compare_retrievers.py`: compares the WildGraphBench retriever and the custom retriever on the WildGraphBench QA benchmark, with optional fixed-generator RAG evaluation
- `tinyLM_LORA.py`: builds retrieved-context QA examples and LoRA fine-tunes TinyLlama
- `lora_with_and_without_trained_embeddings.py`: compares downstream RAG quality with base vs. trained embeddings
- `colab/`: thin Google Colab wrapper notebooks that mount Drive, verify prerequisites, and launch the canonical `.py` entry points without duplicating training logic
- `Lora_trained_vs_untrained_embeddings.ipynb`, `train_embeddings.ipynb`, `transfer_learning.ipynb`, `transfer_learning_mistral.ipynb`: exploratory notebooks for training and comparison experiments
- `WildGraphBench/`: benchmark data and tooling used for retrieval and QA experiments
- `WikiGame/`: a related GraphRAG-style Wikipedia navigation project included in this repo, but separate from the main final-project training pipeline
- `custom-dataset-loader/`: standalone pipeline for building larger custom Wikipedia graph datasets
- `vis/`: saved figures from training and evaluation runs

## Current Progress

Based on the midpoint notes, the project has already completed:

- LoRA fine-tuning of TinyLlama
- fine-tuning of a pretrained retrieval embedding model
- comparison experiments showing stronger performance with fine-tuned embeddings than with the baseline embedder
- an exploratory custom-dataset retriever experiment using `custom-dataset-loader/` artifacts
- a direct comparison between the custom retriever and the existing WildGraphBench retriever on the WildGraphBench QA benchmark

The next planned steps are:

- repeat the workflow on stronger models such as Mistral-class models

The earlier plan to continue scaling experiments on the custom dataset has been retired. After running the first custom retriever experiment, the project direction is to keep `WildGraphBench` as the primary retrieval benchmark and embedding source unless the custom-data pipeline is redesigned in a more task-aligned way.

## Setup

Install the base dependencies:

```bash
pip install -r requirements.txt
```

For the training scripts and notebooks, you will also need the usual Hugging Face and analysis stack used in this repo, including:

- `transformers`
- `datasets`
- `accelerate`
- `sentence-transformers`
- `peft`
- `trl`
- `scikit-learn`
- `matplotlib`
- `beautifulsoup4`

## Colab Wrapper Notebooks

The notebooks under `colab/` exist to support the intended execution workflow for this project:

1. edit code locally in Cursor
2. keep the repository inside a Google Drive-synced folder
3. run training and evaluation from Google Colab against that same synced repo

These notebooks are intentionally a **wrapper layer**, not a second implementation of the pipeline. Their job is to:

- mount Google Drive
- `cd` into the synced repo root
- install any missing dependencies
- verify the required dataset artifacts and checkpoint directories exist
- launch the existing Python scripts with `!python ...`

This keeps the `.py` files as the source of truth and avoids notebook-specific drift or stale interpreter state. In particular:

- `colab/train_custom_retriever.ipynb` wraps `train_custom_retriever.py`
- `colab/compare_retrievers.ipynb` wraps `compare_retrievers.py`

The notebooks assume you are using a Drive-backed copy of this repository. If your generated custom dataset artifacts already exist locally, the preferred path is to sync them into that Drive-backed repo and run the notebooks there. Only rerun the custom dataset pipeline if those generated artifacts are not available.

## Main Workflow

### 1. Fine-tune retrieval embeddings

Use `train_embeddings.py` for a single-domain experiment or `train_embeddings_all_subjects.py` for the full multi-domain benchmark. These scripts start from the pretrained sentence-transformer `all-MiniLM-L6-v2` and then fine-tune it on `WildGraphBench`; they do not train a retrieval model from scratch. They:

- load `WildGraphBench` QA pairs and reference pages
- chunk reference documents
- retrieve positive training chunks for each question
- fine-tune the pretrained sentence-transformer for retrieval
- evaluate retrieval quality during training

### 2. Fine-tune TinyLlama with retrieved context

Run `tinyLM_LORA.py` to:

- retrieve context chunks for each QA example
- format question-context-answer training records
- LoRA fine-tune TinyLlama on that data
- log and plot training/evaluation loss

### 3. Compare baseline vs. trained retrieval

Use `lora_with_and_without_trained_embeddings.py` and the accompanying notebooks/plots in `vis/` to compare:

- retrieval recall
- answer quality under RAG
- training behavior across epochs

### 4. Exploratory custom retriever experiment

After building the custom corpus with `custom-dataset-loader/`, the main repo can train a new SentenceTransformer retriever **without recrawling**. This path was explored as an extension of the main workflow, but it is now documented primarily as a completed experiment rather than an active direction.

Train a custom retriever checkpoint from the custom dataset artifacts:

```bash
python3 train_custom_retriever.py \
  --pages_path custom-dataset-loader/data/processed/pages_sanitized.jsonl \
  --links_path custom-dataset-loader/data/processed/links_table.jsonl \
  --output_path custom_all_embeddings
```

What we tried with the custom dataset:

- reused `custom-dataset-loader/data/processed/pages_sanitized.jsonl` and `custom-dataset-loader/data/processed/links_table.jsonl`
- fine-tuned `sentence-transformers/all-MiniLM-L6-v2` with `train_custom_retriever.py`
- trained on weak supervision from the custom graph structure:

This training path uses weak supervision from the custom dataset itself:
- page title -> page text self-pairs
- hyperlink anchor text -> linked target page text pairs

The output is a new SentenceTransformer checkpoint directory (for example `custom_all_embeddings/`), analogous to the existing WildGraphBench checkpoint `all_embeddings/`.

To compare retrievers on the **existing WildGraphBench QA benchmark** without retraining LoRA:

```bash
python3 compare_retrievers.py \
  --custom_embedder_path ./custom_all_embeddings \
  --wildgraph_embedder_path ./all_embeddings \
  --domain "<existing-domain>" \
  --topic "<existing-topic>"
```

The `domain` and `topic` values must match directories that actually exist under `WildGraphBench/QA/` and `WildGraphBench/corpus/`. The Colab wrapper notebook is designed to inspect the synced benchmark tree and help you choose a valid on-disk pair instead of hardcoding example values.

Optional fixed-generator evaluation with an existing LoRA checkpoint:

```bash
python3 compare_retrievers.py \
  --custom_embedder_path ./custom_all_embeddings \
  --wildgraph_embedder_path ./all_embeddings \
  --domain "<existing-domain>" \
  --topic "<existing-topic>" \
  --lora_checkpoint ./tinyllama-lora-mcu
```

This was the recommended first comparison step before retraining LoRA on any new retriever, because it isolates retriever quality first.

### Outcome of the custom retriever experiment

The first benchmark comparison was run on WildGraphBench QA for `culture / Marvel Cinematic Universe` and showed:

- `Recall@1`: custom `0.28`, WildGraphBench `0.28`
- `Recall@3`: custom `0.46`, WildGraphBench `0.52`
- `Recall@5`: custom `0.60`, WildGraphBench `0.66`

So the custom retriever was competitive at top-1 recall, but it did not outperform the existing WildGraphBench retriever overall.

### Why the custom dataset is not likely to outperform WildGraphBench as-is

The current custom-dataset training setup is not especially well matched to the evaluation task:

- the WildGraphBench embeddings were fine-tuned on supervision designed for graph/RAG-style retrieval on the same benchmark family
- the custom retriever was trained with weaker proxy supervision from page titles and hyperlink anchors rather than benchmark question-to-evidence alignment
- evaluation is based on chunk retrieval for answer-bearing evidence, while the current custom training signal is much closer to page-level semantic relatedness

In other words, the WildGraphBench embeddings are trained with RAG-style retrieval in mind, while the current custom embeddings are trained on a weaker transfer objective. Without major structural changes, the WildGraphBench checkpoint should be expected to remain stronger on the WildGraphBench QA benchmark.

Examples of the kind of changes that would likely be required before expecting the custom dataset to win include:

- chunk-level positive targets instead of whole-page targets
- stronger or cleaner supervision than title/self-pairs and raw anchor links
- harder negatives or more benchmark-aligned retrieval objectives
- possibly hybrid training that combines custom data with benchmark-style supervision

## Integrating `custom-dataset-loader/`

`custom-dataset-loader/` was explored as a possible larger-scale data source for the main experiments. Its job is to generate a training-ready graph dataset; this repository's job is to train retrieval and generation models on top of that data.

At a high level, the integration path that was tested is:

1. Use `custom-dataset-loader/scripts/run_pipeline.py` to build a larger custom Wikipedia dataset.
2. Reuse `custom-dataset-loader/data/processed/pages_sanitized.jsonl` and `custom-dataset-loader/data/processed/links_table.jsonl` as the supervision source for `train_custom_retriever.py`.
3. Save the resulting custom retriever checkpoint separately from `./all_embeddings`.
4. Compare the custom retriever against the existing WildGraphBench retriever with `compare_retrievers.py`.
5. Only after retriever quality is validated, decide whether to retrain LoRA against the new retriever.

That experiment was useful because it demonstrated that the custom dataset can already produce a functioning retriever checkpoint and can be evaluated cleanly against the existing benchmark.

However, this repository is **not** currently continuing experimentation on the custom retriever path. The executive decision is to stop there for now, because the present custom supervision setup is unlikely to beat the WildGraphBench embeddings without a more substantial redesign of the training objective and data structure.

In other words, `WildGraphBench` remains the current benchmark, the current best-performing retriever source, and the main basis for downstream RAG and LoRA experiments in this project. `custom-dataset-loader/` remains a useful exploratory extension, but not the active experimental priority.

For dataset-building details, commands, and artifact descriptions, see `custom-dataset-loader/README.md`.

## References

- `https://arxiv.org/pdf/2602.02053#cite.edge2025localglobalgraphrag`
- `https://cs229.stanford.edu/proj2015/309_report.pdf`
- `https://arxiv.org/html/2511.10585v1`
- `https://aclanthology.org/2025.clicit-1.63.pdf`
- `https://microsoft.github.io/graphrag/`
- `https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/`
- `https://github.com/run-llama/llama_index`
- `https://arxiv.org/pdf/2404.16130`
- `https://developers.llamaindex.ai/python/examples/cookbooks/graphrag_v2/`
- `https://developers.llamaindex.ai/python/examples/query_engine/knowledge_graph_rag_query_engine/`

