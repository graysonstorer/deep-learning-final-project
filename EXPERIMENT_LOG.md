# Experiment Log

This file is a brief running log of custom retriever training experiments and comparisons against the existing WildGraphBench retriever. The goal is to track custom-dataset training methodologies over time and measure whether they can eventually outperform the WildGraphBench-trained baseline on benchmark retrieval.

## Entry 1: Custom Weak-Supervision Retriever vs WildGraphBench Retriever

### Objective
Train a SentenceTransformer retriever on the generated custom dataset artifacts, then compare it against the existing WildGraphBench retriever on the WildGraphBench QA benchmark.

### Custom Training Methodology
- Base model: `sentence-transformers/all-MiniLM-L6-v2`
- Training script: `train_custom_retriever.py`
- Data source: `custom-dataset-loader/data/processed/pages_sanitized.jsonl` and `custom-dataset-loader/data/processed/links_table.jsonl`
- Supervision used:
  - page title -> page text
  - hyperlink anchor text -> linked target page text
- Loss: `MultipleNegativesRankingLoss`
- Validation: internal information-retrieval evaluation on a held-out split of the custom weak-supervision pairs

This training method has value now because it proves the custom dataset can already support retriever fine-tuning without recrawling and without retraining LoRA. It also provides a clean first transfer-learning baseline from custom Wikipedia structure into the existing benchmark pipeline.

### Result Summary
- Benchmark: WildGraphBench QA, `culture / Marvel Cinematic Universe`
- Comparison artifact: `vis/custom_vs_wildgraph_retriever_recall_comparison.csv`

| Metric | Custom Retriever | WildGraphBench Retriever |
|---|---:|---:|
| Recall@1 | 0.28 | 0.28 |
| Recall@3 | 0.46 | 0.52 |
| Recall@5 | 0.60 | 0.66 |

### Interpretation
The WildGraphBench retriever still outperforms the custom retriever on this benchmark. The most likely reason is that the WildGraphBench retriever was trained on supervision that is much more closely aligned with the evaluation task: benchmark questions and benchmark reference chunks. The custom retriever was trained with weaker supervision from titles and hyperlink anchors, which is useful for learning semantic relatedness but is not as directly optimized for retrieving answer-bearing benchmark chunks.

The result is still useful:
- the custom retriever is competitive at `Recall@1`
- the custom dataset is already capable of producing a working fine-tuned retriever checkpoint
- the current gap gives a concrete baseline for future custom-training improvements

### Visuals
- Training dynamics: `vis/custom_retriever_training_dynamics.png`
- Retriever comparison: `vis/custom_vs_wildgraph_retriever_recall_comparison.png`

### Takeaway
At this stage, custom weak-supervision training is a valid and valuable first retriever adaptation step, but benchmark-aligned WildGraphBench training still performs better on the WildGraphBench QA task.

---

## Template For Future Entries

## Entry N: <short experiment name>

### Objective
<What changed, and what was being tested?>

### Custom Training Methodology
- Base model:
- Training script / notebook:
- Dataset artifacts used:
- Supervision used:
- Key hyperparameters:
- Any changes from the previous method:

### Result Summary
- Benchmark:
- Comparison artifact(s):

| Metric | Custom Retriever | WildGraphBench Retriever | Notes |
|---|---:|---:|---|
| Recall@1 |  |  |  |
| Recall@3 |  |  |  |
| Recall@5 |  |  |  |

### Interpretation
<Why did this method help or not help?>

### Visuals
- Training dynamics:
- Retriever comparison:
- Optional downstream QA:

### Takeaway
<One short conclusion to carry forward>

---

## Roadmap To Outperform WildGraphBench Embeddings

This roadmap is the planned experimental ladder for improving the custom retriever while preserving clean comparisons against the existing WildGraphBench retriever.

### Phase 1: Broaden Evaluation Before Changing Training

Evaluate the current checkpoints across all available WildGraphBench domain/topic pairs:

- base `sentence-transformers/all-MiniLM-L6-v2`
- `all_embeddings/`
- `custom_all_embeddings/`

Collect at least:

- `Recall@1`
- `Recall@3`
- `Recall@5`

Purpose:

- determine whether the custom retriever is behind everywhere or only on some benchmark slices
- identify whether the current result on `culture / Marvel Cinematic Universe` is representative
- establish a stronger baseline before changing the training methodology

### Phase 2: Run A True Generalization Test

Do not use the current `all_embeddings/` checkpoint as evidence of unseen-domain generalization, because it was trained across the WildGraphBench domains.

Instead:

1. choose one or two target WildGraphBench domains to hold out
2. train a WildGraphBench retriever excluding those target domains
3. compare the held-out WildGraphBench retriever against:
   - the base model
   - the current custom retriever

Purpose:

- test whether the custom dataset leads to better transfer to unseen benchmark domains
- separate in-domain benchmark advantage from true generalization behavior

### Phase 3: Low-Cost Custom Retraining Changes

Before redesigning the training method, rerun the current custom retriever pipeline with more supervision from the existing link structure.

Priority change:

- increase `--max_edges` substantially beyond the current default if runtime permits

Purpose:

- get more training signal without changing the architecture or objective
- test whether the current method is underusing available weak supervision

### Phase 4: Improve Task Alignment With Chunk-Level Training

If the current custom method remains behind, move from full-page positives to chunk-level positives.

Recommended direction:

- chunk custom pages into retrieval-sized text chunks
- continue using title and anchor text as queries
- train against target-page chunks rather than full-page text blobs

Purpose:

- better match the actual evaluation setup, which is chunk retrieval rather than whole-page retrieval
- improve ranking of answer-bearing evidence within the top retrieved results

### Phase 5: Improve Supervision Quality

If chunk-level training still does not close the gap, refine the weak supervision itself.

Candidate improvements:

- filter weak or generic anchors
- rebalance title self-pairs vs anchor-based pairs
- introduce harder negatives beyond in-batch negatives

Purpose:

- reduce noisy supervision
- make the custom retriever focus more directly on discriminative retrieval behavior

### Phase 6: Only Then Consider Larger Changes

After the earlier phases are complete, consider more ambitious approaches such as:

- graph-aware positives or multi-hop supervision
- combined custom + WildGraphBench retriever training
- downstream LoRA retraining with the improved retriever

Purpose:

- defer higher-complexity work until the simpler, more interpretable retrieval experiments are exhausted

### Recommended Near-Term Order

1. evaluate current checkpoints across all WildGraphBench domains
2. document per-domain and average results in this log
3. run a held-out-domain WildGraphBench baseline for a true generalization test
4. rerun custom training with a larger `--max_edges`
5. move to chunk-level custom retriever training
6. refine supervision quality
7. only then revisit larger architectural or downstream training changes
