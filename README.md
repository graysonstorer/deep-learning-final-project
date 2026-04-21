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

The next planned steps are:

- train the retrieval and LoRA pipelines on the larger custom dataset
- repeat the workflow on stronger models such as Mistral-class models

## Setup

Install the base dependencies:

```bash
pip install -r requirements.txt
```

For the training scripts and notebooks, you will also need the usual Hugging Face and analysis stack used in this repo, including:

- `transformers`
- `datasets`
- `accelerate`
- `peft`
- `trl`
- `scikit-learn`
- `matplotlib`
- `beautifulsoup4`

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

### 4. Train and compare a custom retriever

After building the custom corpus with `custom-dataset-loader/`, the main repo can train a new SentenceTransformer retriever **without recrawling**. This path is separate from the original WildGraphBench workflow and does not modify it.

Train a custom retriever checkpoint from the custom dataset artifacts:

```bash
python3 train_custom_retriever.py \
  --pages_path custom-dataset-loader/data/processed/pages_sanitized.jsonl \
  --links_path custom-dataset-loader/data/processed/links_table.jsonl \
  --output_path custom_all_embeddings
```

This training path uses weak supervision from the custom dataset itself:
- page title -> page text self-pairs
- hyperlink anchor text -> linked target page text pairs

The output is a new SentenceTransformer checkpoint directory (for example `custom_all_embeddings/`), analogous to the existing WildGraphBench checkpoint `all_embeddings/`.

To compare retrievers on the **existing WildGraphBench QA benchmark** without retraining LoRA:

```bash
python3 compare_retrievers.py \
  --custom_embedder_path ./custom_all_embeddings \
  --wildgraph_embedder_path ./all_embeddings \
  --domain culture \
  --topic "Marvel Cinematic Universe"
```

Optional fixed-generator evaluation with an existing LoRA checkpoint:

```bash
python3 compare_retrievers.py \
  --custom_embedder_path ./custom_all_embeddings \
  --wildgraph_embedder_path ./all_embeddings \
  --domain culture \
  --topic "Marvel Cinematic Universe" \
  --lora_checkpoint ./tinyllama-lora-mcu
```

This is the recommended first comparison step before retraining LoRA on any new retriever.

## Integrating `custom-dataset-loader/`

`custom-dataset-loader/` is intended to become the larger-scale data source for the main experiments. Its job is to generate a training-ready graph dataset; this repository's job is to train retrieval and generation models on top of that data.

At a high level, the current integration path is:

1. Use `custom-dataset-loader/scripts/run_pipeline.py` to build a larger custom Wikipedia dataset.
2. Reuse `custom-dataset-loader/data/processed/pages_sanitized.jsonl` and `custom-dataset-loader/data/processed/links_table.jsonl` as the supervision source for `train_custom_retriever.py`.
3. Save the resulting custom retriever checkpoint separately from `./all_embeddings`.
4. Compare the custom retriever against the existing WildGraphBench retriever with `compare_retrievers.py`.
5. Only after retriever quality is validated, decide whether to retrain LoRA against the new retriever.

In other words, `WildGraphBench` is the current benchmark and prototype dataset, while `custom-dataset-loader/` is the planned path toward a larger, more general training corpus.

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

