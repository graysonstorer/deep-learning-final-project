from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from compare_retrievers import (
    DEFAULT_WILDBENCH_ROOT,
    evaluate_rag,
    load_lora_model,
    load_questions,
    load_reference_pages,
    resolve_lora_checkpoint,
    retrieval_recall_at_k,
)


def sanitize_label(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "model"


def load_loss_history(adapter_root: Path) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    checkpoint_path = resolve_lora_checkpoint(adapter_root)
    trainer_state_path = checkpoint_path / "trainer_state.json"
    if not trainer_state_path.exists():
        return [], []

    with trainer_state_path.open("r", encoding="utf-8") as f:
        trainer_state = json.load(f)

    log_history = trainer_state.get("log_history", [])
    train_loss = [
        (float(entry["epoch"]), float(entry["loss"]))
        for entry in log_history
        if "loss" in entry and "epoch" in entry
    ]
    eval_loss = [
        (float(entry["epoch"]), float(entry["eval_loss"]))
        for entry in log_history
        if "eval_loss" in entry and "epoch" in entry
    ]
    return train_loss, eval_loss


def plot_model_comparison(
    metrics_rows: Sequence[Dict[str, object]],
    loss_histories: Dict[str, Dict[str, List[Tuple[float, float]]]],
    plot_path: Path,
    title: str,
    retrieval_label: str,
    retrieval_recall: float,
) -> None:
    has_eval_loss = any(history["eval"] for history in loss_histories.values())
    if has_eval_loss:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        metric_ax = axes[0]
        loss_ax = axes[1]
    else:
        fig, metric_ax = plt.subplots(1, 1, figsize=(7, 5))
        loss_ax = None

    labels = [str(row["model_label"]) for row in metrics_rows]
    gen_f1_scores = [float(row["gen_f1"]) for row in metrics_rows]
    x = np.arange(len(labels))

    bars = metric_ax.bar(x, gen_f1_scores, color=["steelblue", "darkorange"][: len(labels)])
    metric_ax.set_xticks(x)
    metric_ax.set_xticklabels(labels)
    metric_ax.set_ylim(0, min(1.0, max(0.1, max(gen_f1_scores, default=0.0) * 1.2)))
    metric_ax.set_ylabel("Generation F1")
    metric_ax.set_title("Fine-Tuned Model Comparison")
    metric_ax.grid(axis="y", alpha=0.3)
    for bar, score in zip(bars, gen_f1_scores):
        metric_ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )

    metric_ax.text(
        0.02,
        0.96,
        f"Shared retriever: {retrieval_label}\nRecall@{metrics_rows[0]['k']}: {retrieval_recall:.3f}",
        transform=metric_ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    if loss_ax is not None:
        for model_label, history in loss_histories.items():
            eval_loss = history["eval"]
            if not eval_loss:
                continue
            epochs, values = zip(*eval_loss)
            loss_ax.plot(epochs, values, marker="o", label=f"{model_label} eval")
        loss_ax.set_title("Eval Loss by Epoch")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Eval Loss")
        loss_ax.grid(True)
        loss_ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved model comparison plot to: {plot_path}")


def write_metrics_csv(rows: Sequence[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_label",
        "checkpoint_root",
        "base_model_name",
        "retriever_label",
        "retriever_path",
        "retrieval_recall",
        "gen_f1",
        "domain",
        "topic",
        "k",
        "limit",
        "n_gen",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved comparison metrics to: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Directly compare two fine-tuned LoRA chat models on the same WildGraphBench slice.")
    parser.add_argument("--repo_path", type=str, default=str(DEFAULT_WILDBENCH_ROOT))
    parser.add_argument("--domain", type=str, default="culture")
    parser.add_argument("--topic", type=str, default="Marvel Cinematic Universe")
    parser.add_argument("--embedder_path", type=str, default="./all_embeddings")
    parser.add_argument("--embedder_label", type=str, default="wildgraph")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--word_coverage_threshold", type=float, default=0.5)
    parser.add_argument("--n_gen", type=int, default=20)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--model_a_label", type=str, default="TinyLlama")
    parser.add_argument("--model_a_checkpoint", type=str, default="./tinyllama-lora-mcu")
    parser.add_argument("--model_a_base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--model_b_label", type=str, default="Llama-3.2-3B")
    parser.add_argument("--model_b_checkpoint", type=str, default="./llama32-3b-lora-mcu")
    parser.add_argument("--model_b_base_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_plot", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_path = Path(args.repo_path)
    qa_path = repo_path / "QA" / args.domain / "questions.jsonl"
    questions, gold_answers = load_questions(qa_path)
    questions = questions[: args.limit]
    gold_answers = gold_answers[: args.limit]
    all_chunks = load_reference_pages(repo_path, args.domain, args.topic, chunk_size=args.chunk_size)
    all_chunk_texts = [chunk_text for _, chunk_text in all_chunks]

    embedder = SentenceTransformer(args.embedder_path)
    retrieval_recall = retrieval_recall_at_k(
        questions,
        gold_answers,
        all_chunk_texts,
        embedder,
        k=args.k,
        word_coverage_threshold=args.word_coverage_threshold,
    )

    print(f"Evaluating on {len(questions)} QA examples from domain={args.domain!r}, topic={args.topic!r}")
    print(f"Reference chunks: {len(all_chunk_texts)}")
    print(f"Shared retriever [{args.embedder_label}] Recall@{args.k}: {retrieval_recall:.3f}")
    print("")

    model_specs = [
        {
            "label": args.model_a_label,
            "checkpoint": Path(args.model_a_checkpoint),
            "base_model": args.model_a_base_model,
        },
        {
            "label": args.model_b_label,
            "checkpoint": Path(args.model_b_checkpoint),
            "base_model": args.model_b_base_model,
        },
    ]

    metrics_rows: List[Dict[str, object]] = []
    loss_histories: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    for spec in model_specs:
        print(f"=== Evaluating {spec['label']} ===")
        model, tokenizer = load_lora_model(
            adapter_root=spec["checkpoint"],
            base_model_name=spec["base_model"],
            device=args.device,
        )
        metrics = evaluate_rag(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            gold_answers=gold_answers,
            all_chunk_texts=all_chunk_texts,
            embed_model=embedder,
            label=str(spec["label"]),
            k=args.k,
            n_gen=args.n_gen,
        )
        train_loss, eval_loss = load_loss_history(spec["checkpoint"])
        loss_histories[str(spec["label"])] = {"train": train_loss, "eval": eval_loss}
        metrics_rows.append(
            {
                "model_label": spec["label"],
                "checkpoint_root": str(spec["checkpoint"]),
                "base_model_name": spec["base_model"],
                "retriever_label": args.embedder_label,
                "retriever_path": args.embedder_path,
                "retrieval_recall": retrieval_recall,
                "gen_f1": metrics["gen_f1"],
                "domain": args.domain,
                "topic": args.topic,
                "k": args.k,
                "limit": args.limit,
                "n_gen": args.n_gen,
            }
        )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print("")

    default_stub = f"{sanitize_label(args.model_a_label)}_vs_{sanitize_label(args.model_b_label)}"
    csv_path = Path(args.output_csv) if args.output_csv else Path("vis") / f"{default_stub}_comparison.csv"
    plot_path = Path(args.output_plot) if args.output_plot else Path("vis") / f"{default_stub}_comparison.png"

    write_metrics_csv(metrics_rows, csv_path)
    plot_model_comparison(
        metrics_rows=metrics_rows,
        loss_histories=loss_histories,
        plot_path=plot_path,
        title=f"Fine-Tuned Model Comparison on {args.domain} / {args.topic}",
        retrieval_label=args.embedder_label,
        retrieval_recall=retrieval_recall,
    )


if __name__ == "__main__":
    main()
