from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_WILDBENCH_ROOT = REPO_ROOT / "WildGraphBench"


def load_questions(path: Path) -> Tuple[List[str], List[str]]:
    questions, gold_answers = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "answer" not in item:
                continue
            questions.append(item["question"])
            gold_answers.append(item["answer"])
    return questions, gold_answers


def load_reference_pages(repo_root: Path, domain: str, topic: str, chunk_size: int = 300) -> List[Tuple[str, str]]:
    folder = repo_root / "corpus" / domain / topic / "reference_pages"
    all_chunks: List[Tuple[str, str]] = []
    for filepath in glob.glob(str(folder / "*.txt")):
        with open(filepath, "r", errors="ignore") as f:
            text = f.read()
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            node_id = f"{Path(filepath).name}__{i}"
            all_chunks.append((node_id, chunk))
    return all_chunks


def retrieval_recall_at_k(
    questions: List[str],
    gold_answers: List[str],
    all_chunk_texts: List[str],
    embed_model: SentenceTransformer,
    k: int = 3,
    word_coverage_threshold: float = 0.5,
) -> float:
    stopwords = {
        "that",
        "with",
        "this",
        "have",
        "from",
        "they",
        "been",
        "were",
        "their",
        "there",
        "when",
        "which",
        "will",
        "would",
        "could",
        "should",
        "about",
        "into",
        "than",
        "then",
        "also",
        "some",
    }

    chunk_embeddings = embed_model.encode(all_chunk_texts, convert_to_tensor=True, show_progress_bar=False)
    hits = 0

    for question, answer in zip(questions, gold_answers):
        q_emb = embed_model.encode(question, convert_to_tensor=True)
        sims = util.cos_sim(q_emb, chunk_embeddings)[0]
        top_indices = sims.topk(k).indices.tolist()
        top_chunks = " ".join(all_chunk_texts[i] for i in top_indices).lower()

        answer_lower = answer.lower().strip()
        exact_hit = answer_lower in top_chunks
        answer_words = [w for w in answer_lower.split() if len(w) > 3 and w not in stopwords]
        coverage_hit = False
        if answer_words:
            coverage = sum(1 for w in answer_words if w in top_chunks) / len(answer_words)
            coverage_hit = coverage >= word_coverage_threshold

        if exact_hit or coverage_hit:
            hits += 1

    return hits / max(len(questions), 1)


def answer_f1(prediction: str, gold: str) -> float:
    pred_tokens = set(prediction.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    context: str,
    max_new_tokens: int = 100,
) -> str:
    prompt = f"""<|system|>
You are a helpful assistant answering questions about the Marvel Cinematic Universe.
</s>
<|user|>
Context:
{context}

Question: {question}
</s>
<|assistant|>
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return generated.strip()


def evaluate_rag(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    questions: List[str],
    gold_answers: List[str],
    all_chunk_texts: List[str],
    embed_model: SentenceTransformer,
    label: str,
    k: int,
    n_gen: int,
) -> Dict[str, float]:
    recall = retrieval_recall_at_k(questions, gold_answers, all_chunk_texts, embed_model, k=k)

    subset = list(zip(questions, gold_answers))[:n_gen]
    chunk_embeddings = embed_model.encode(all_chunk_texts, convert_to_tensor=True, show_progress_bar=False)
    f1_scores: List[float] = []
    for q, gold in subset:
        q_emb = embed_model.encode(q, convert_to_tensor=True)
        sims = util.cos_sim(q_emb, chunk_embeddings)[0]
        top_indices = sims.topk(k).indices.tolist()
        context = "\n\n".join(all_chunk_texts[i] for i in top_indices)
        pred = generate_answer(model, tokenizer, q, context)
        f1_scores.append(answer_f1(pred, gold))

    gen_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    print(f"  [{label}] Recall@{k}: {recall:.3f} | Gen F1: {gen_f1:.3f}")
    return {"recall": recall, "gen_f1": gen_f1}


def resolve_lora_checkpoint(path: Path) -> Path:
    if (path / "adapter_config.json").exists():
        return path

    checkpoints = sorted(
        [p for p in path.glob("checkpoint-*") if (p / "adapter_config.json").exists()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No PEFT adapter checkpoint found under {path}")
    return checkpoints[-1]


def load_lora_model(
    adapter_root: Path,
    base_model_name: str,
    device: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    adapter_path = resolve_lora_checkpoint(adapter_root)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.to(device)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare WildGraphBench and custom retrievers on the WildGraphBench QA benchmark.")
    parser.add_argument("--repo_path", type=str, default=str(DEFAULT_WILDBENCH_ROOT))
    parser.add_argument("--domain", type=str, default="culture")
    parser.add_argument("--topic", type=str, default="Marvel Cinematic Universe")
    parser.add_argument("--wildgraph_embedder_path", type=str, default="./all_embeddings")
    parser.add_argument("--custom_embedder_path", type=str, required=True)
    parser.add_argument("--include_base", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=50, help="Number of QA examples to evaluate")
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--word_coverage_threshold", type=float, default=0.5)
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Optional PEFT checkpoint root for fixed-generator RAG evaluation")
    parser.add_argument("--base_model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--n_gen", type=int, default=20)
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    repo_path = Path(args.repo_path)
    qa_path = repo_path / "QA" / args.domain / "questions.jsonl"
    questions, gold_answers = load_questions(qa_path)
    questions = questions[: args.limit]
    gold_answers = gold_answers[: args.limit]

    all_chunks = load_reference_pages(repo_path, args.domain, args.topic, chunk_size=args.chunk_size)
    all_chunk_texts = [c[1] for c in all_chunks]

    embedders: List[Tuple[str, SentenceTransformer]] = []
    if args.include_base:
        embedders.append(("base", SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")))
    embedders.append(("wildgraph", SentenceTransformer(args.wildgraph_embedder_path)))
    embedders.append(("custom", SentenceTransformer(args.custom_embedder_path)))

    print(f"Evaluating on {len(questions)} QA examples from domain={args.domain!r}, topic={args.topic!r}")
    print(f"Reference chunks: {len(all_chunk_texts)}")
    print("")

    print("=== Retriever Comparison ===")
    retrieval_results: Dict[str, float] = {}
    for label, embedder in embedders:
        recall = retrieval_recall_at_k(
            questions,
            gold_answers,
            all_chunk_texts,
            embedder,
            k=args.k,
            word_coverage_threshold=args.word_coverage_threshold,
        )
        retrieval_results[label] = recall
        print(f"{label:10s} Recall@{args.k}: {recall:.3f}")

    if args.lora_checkpoint:
        print("\n=== Fixed-Generator RAG Comparison ===")
        model, tokenizer = load_lora_model(
            adapter_root=Path(args.lora_checkpoint),
            base_model_name=args.base_model_name,
            device=args.device,
        )
        for label, embedder in embedders:
            evaluate_rag(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                gold_answers=gold_answers,
                all_chunk_texts=all_chunk_texts,
                embed_model=embedder,
                label=label,
                k=args.k,
                n_gen=args.n_gen,
            )


if __name__ == "__main__":
    main()

