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
from datasets import Dataset

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


def compute_token_accuracy(model, tokenizer, dataset, device, max_samples=50):
    """
    Compute token-level accuracy on a dataset.
    Adapted from the training script's compute_token_accuracy function.
    """
    model.eval()
    correct, total = 0, 0
    samples = dataset.select(range(min(max_samples, len(dataset))))
    
    # Set pad_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    for example in samples:
        inputs = tokenizer(
            example["text"],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,  # Don't pad individual examples
        ).to(device)
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)
        
        # Shift: predict token[t+1] from logits[t]
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        predictions = shift_logits.argmax(dim=-1)
        
        # Create mask: exclude pad tokens and special tokens if needed
        # For most tokenizers, we want to evaluate on all tokens except padding
        if tokenizer.pad_token_id is not None:
            mask = (shift_labels != tokenizer.pad_token_id)
        else:
            # If no pad token, evaluate on all tokens
            mask = torch.ones_like(shift_labels, dtype=torch.bool)
        
        # Ensure mask is a tensor (not a scalar bool)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
        
        correct += ((predictions == shift_labels) & mask).sum().item()
        total += mask.sum().item()
    
    return correct / total if total > 0 else 0.0



def get_positive_chunks(questions, gold_answers, all_chunks_with_ids, embed_model, top_k=3):
    """
    Retrieve positive chunks for questions using the embedder.
    Adapted from the training script's get_positive_chunks function.
    """
    from sentence_transformers import util
    
    all_chunk_texts = [c[1] for c in all_chunks_with_ids]
    chunk_embeddings = embed_model.encode(all_chunk_texts, convert_to_tensor=True)
    retrieved_positives = []

    for question, answer in zip(questions, gold_answers):
        answer_words = set(answer.lower().split())
        scored = []
        for chunk_text in all_chunk_texts:
            chunk_words = set(chunk_text.lower().split())
            overlap = len(answer_words & chunk_words) / max(len(answer_words), 1)
            scored.append(overlap)

        best_overlap = max(scored)
        if best_overlap > 0.3:
            top_indices = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)[:top_k]
        else:
            answer_emb = embed_model.encode(answer, convert_to_tensor=True)
            sims = util.cos_sim(answer_emb, chunk_embeddings)[0]
            top_indices = sims.topk(top_k).indices.tolist()

        retrieved_positives.append([all_chunk_texts[i] for i in top_indices])
    return retrieved_positives


def make_lora_dataset(questions, gold_answers, retrieved_positives):
    """
    Create a dataset for token accuracy evaluation.
    Adapted from the training script's make_lora_dataset function.
    """
    records = []
    for q, answer, pos_chunks in zip(questions, gold_answers, retrieved_positives):
        context = "\n\n".join(pos_chunks)
        text = f"""<|system|>
You are a helpful assistant answering questions about the Marvel Cinematic Universe.
</s>
<|user|>
Context:
{context}

Question: {q}
</s>
<|assistant|>
{answer}</s>"""
        records.append({"text": text})
    return Dataset.from_list(records)


def plot_model_comparison(
    metrics_rows: Sequence[Dict[str, object]],
    loss_histories: Dict[str, Dict[str, List[Tuple[float, float]]]],
    plot_path: Path,
    title: str,
    retrieval_label: str,
    retrieval_recall: float,
) -> None:
    has_eval_loss = any(history["eval"] for history in loss_histories.values())
    
    # Now we have 3 metrics to plot: gen_f1, token_accuracy, and optionally eval_loss
    if has_eval_loss:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metric_ax = axes[0]
        token_ax = axes[1]
        loss_ax = axes[2]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        metric_ax = axes[0]
        token_ax = axes[1]
        loss_ax = None

    labels = [str(row["model_label"]) for row in metrics_rows]
    gen_f1_scores = [float(row["gen_f1"]) for row in metrics_rows]
    token_acc_scores = [float(row["token_accuracy"]) for row in metrics_rows]
    x = np.arange(len(labels))

    # Plot 1: Generation F1
    bars = metric_ax.bar(x, gen_f1_scores, color=["steelblue", "darkorange"][: len(labels)])
    metric_ax.set_xticks(x)
    metric_ax.set_xticklabels(labels)
    metric_ax.set_ylim(0, min(1.0, max(0.1, max(gen_f1_scores, default=0.0) * 1.2)))
    metric_ax.set_ylabel("Generation F1")
    metric_ax.set_title("Generation F1 Comparison")
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

    # Plot 2: Token Accuracy
    bars_token = token_ax.bar(x, token_acc_scores, color=["forestgreen", "crimson"][: len(labels)])
    token_ax.set_xticks(x)
    token_ax.set_xticklabels(labels)
    token_ax.set_ylim(0, min(1.0, max(0.1, max(token_acc_scores, default=0.0) * 1.2)))
    token_ax.set_ylabel("Token Accuracy")
    token_ax.set_title("Token Accuracy Comparison")
    token_ax.grid(axis="y", alpha=0.3)
    for bar, score in zip(bars_token, token_acc_scores):
        token_ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )

    # Plot 3: Eval Loss (if available)
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
        "token_accuracy",
        "domain",
        "topic",
        "k",
        "limit",
        "n_gen",
        "token_acc_samples",
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
    parser.add_argument("--token_acc_samples", type=int, default=50, help="Number of samples for token accuracy evaluation")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--model_a_label", type=str, default="TinyLlama")
    parser.add_argument("--model_a_checkpoint", type=str, default="./tinyllama-lora-mcu")
    parser.add_argument("--model_a_base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--model_b_label", type=str, default="Llama-3.2-3B")
    parser.add_argument("--model_b_checkpoint", type=str, default="./llama32-3b-lora-mcu")
    parser.add_argument("--model_b_base_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--model_c_label", type = str, default = None)
    parser.add_argument("--model_c_checkpoint", type = str, default = None)
    parser.add_argument("--model_c_base_model", type = str, default = None)
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

    # Prepare evaluation dataset for token accuracy
    # We'll create it once and reuse for both models
    print("Preparing token accuracy evaluation dataset...")
    retrieved_positives = get_positive_chunks(
        questions, gold_answers, all_chunks, embedder, top_k=args.k
    )
    eval_dataset = make_lora_dataset(questions, gold_answers, retrieved_positives)
    print(f"Token accuracy dataset size: {len(eval_dataset)} examples")
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
        {
            "label": args.model_c_label,
            "checkpoint": Path(args.model_c_checkpoint),
            "base_model": args.model_c_base_model,
        }
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
        
        # Compute RAG metrics (recall and generation F1)
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
        
        # Compute token accuracy
        print(f"Computing token accuracy on {args.token_acc_samples} samples...")
        token_accuracy = compute_token_accuracy(
            model=model,
            tokenizer=tokenizer,
            dataset=eval_dataset,
            device=args.device,
            max_samples=args.token_acc_samples,
        )
        print(f"Token Accuracy: {token_accuracy:.3f}")
        
        # Load loss history
        train_loss, eval_loss = load_loss_history(spec["checkpoint"])
        loss_histories[str(spec["label"])] = {"train": train_loss, "eval": eval_loss}
        
        # Store all metrics
        metrics_rows.append(
            {
                "model_label": spec["label"],
                "checkpoint_root": str(spec["checkpoint"]),
                "base_model_name": spec["base_model"],
                "retriever_label": args.embedder_label,
                "retriever_path": args.embedder_path,
                "retrieval_recall": retrieval_recall,
                "gen_f1": metrics["gen_f1"],
                "token_accuracy": token_accuracy,  # Added token accuracy
                "domain": args.domain,
                "topic": args.topic,
                "k": args.k,
                "limit": args.limit,
                "n_gen": args.n_gen,
                "token_acc_samples": args.token_acc_samples,  # Track sample count
            }
        )
        
        # Clean up
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