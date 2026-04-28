from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from chat_prompting import prepare_tokenizer, render_chat_prompt


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


def load_reference_pages(repo_path: Path, domain: str, topic: str, chunk_size: int = 300) -> List[Tuple[str, str]]:
    folder = repo_path / "corpus" / domain / topic / "reference_pages"
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


def get_positive_chunks(
    questions: Sequence[str],
    gold_answers: Sequence[str],
    all_chunks_with_ids: Sequence[Tuple[str, str]],
    embed_model: SentenceTransformer,
    top_k: int = 3,
    overlap_threshold: float = 0.3,
) -> List[List[str]]:
    all_chunk_texts = [chunk_tuple[1] for chunk_tuple in all_chunks_with_ids]
    chunk_embeddings = embed_model.encode(all_chunk_texts, convert_to_tensor=True)
    retrieved_positives: List[List[str]] = []

    for answer in gold_answers:
        answer_words = set(answer.lower().split())
        scored = []
        for chunk_text in all_chunk_texts:
            chunk_words = set(chunk_text.lower().split())
            overlap = len(answer_words & chunk_words) / max(len(answer_words), 1)
            scored.append(overlap)

        best_overlap = max(scored)

        if best_overlap > overlap_threshold:
            top_indices = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)[:top_k]
        else:
            answer_emb = embed_model.encode(answer, convert_to_tensor=True)
            sims = util.cos_sim(answer_emb, chunk_embeddings)[0]
            top_indices = sims.topk(top_k).indices.tolist()

        retrieved_positives.append([all_chunk_texts[i] for i in top_indices])

    return retrieved_positives


def make_lora_dataset(
    tokenizer: AutoTokenizer,
    questions: Sequence[str],
    gold_answers: Sequence[str],
    retrieved_positives: Sequence[Sequence[str]],
) -> Dataset:
    records = []
    for question, answer, pos_chunks in zip(questions, gold_answers, retrieved_positives):
        context = "\n\n".join(pos_chunks)
        text = render_chat_prompt(
            tokenizer=tokenizer,
            question=question,
            context=context,
            answer=answer,
        )
        records.append({"text": text})
    return Dataset.from_list(records)


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested_device


def resolve_precision(device: str) -> Tuple[torch.dtype, bool, bool]:
    if device != "cuda":
        return torch.float32, False, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False, True
    return torch.float16, True, False


def verify_target_modules(model: torch.nn.Module, target_modules: Sequence[str]) -> None:
    found = set()
    for module_name, _ in model.named_modules():
        for target in target_modules:
            if module_name.endswith(target):
                found.add(target)
    missing = sorted(set(target_modules) - found)
    if missing:
        raise ValueError(f"LoRA target modules not found on model: {missing}")


def run_preflight(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    sample_text: str,
    device: str,
    max_length: int,
) -> None:
    preview = sample_text[:400].replace("\n", "\\n")
    print(f"Prompt preview: {preview}...")

    encoded = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    print(f"Sample tokenized length: {encoded['input_ids'].shape[-1]}")

    if device == "cuda":
        encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}

    model.eval()
    with torch.no_grad():
        model(**encoded)
    model.train()
    print("Preflight forward pass succeeded.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Llama 3.2 3B on WildGraphBench QA examples.")
    parser.add_argument("--repo_path", type=str, default="WildGraphBench")
    parser.add_argument("--domain", type=str, default="culture")
    parser.add_argument("--topic", type=str, default="Marvel Cinematic Universe")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--embedder_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", type=str, default="llama32-3b-lora-mcu")
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--overlap_threshold", type=float, default=0.3)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--train_test_split", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.45)
    parser.add_argument("--preflight_max_length", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch_dtype, fp16, bf16 = resolve_precision(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Using device: {device}")
    print(f"Using dtype: {torch_dtype}")

    tokenizer = prepare_tokenizer(
        AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    )
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    target_modules = [part.strip() for part in args.target_modules.split(",") if part.strip()]
    verify_target_modules(model, target_modules)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if device == "cuda":
        model.to(device)

    repo_path = Path(args.repo_path)
    qa_path = repo_path / "QA" / args.domain / "questions.jsonl"
    questions, gold_answers = load_questions(qa_path)
    all_chunks = load_reference_pages(repo_path, args.domain, args.topic, chunk_size=args.chunk_size)

    embed_model = SentenceTransformer(args.embedder_name)
    retrieved_positives = get_positive_chunks(
        questions,
        gold_answers,
        all_chunks,
        embed_model,
        top_k=args.top_k,
        overlap_threshold=args.overlap_threshold,
    )

    dataset = make_lora_dataset(tokenizer, questions, gold_answers, retrieved_positives)
    print(f"Dataset size: {len(dataset)} examples")
    print("\nSample:")
    print(dataset[0]["text"])

    run_preflight(
        model=model,
        tokenizer=tokenizer,
        sample_text=dataset[0]["text"],
        device=device,
        max_length=args.preflight_max_length,
    )

    dataset = dataset.train_test_split(test_size=args.train_test_split, seed=args.seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=fp16,
        bf16=bf16,
        use_cpu=(device == "cpu"),
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        load_best_model_at_end=False,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        formatting_func=lambda example: example["text"],
    )
    trainer.train()

    log_history = trainer.state.log_history
    train_loss = [(entry["epoch"], entry["loss"]) for entry in log_history if "loss" in entry]
    eval_loss = [(entry["epoch"], entry["eval_loss"]) for entry in log_history if "eval_loss" in entry]

    if train_loss and eval_loss:
        plt.figure(figsize=(7, 4))
        plt.plot(*zip(*train_loss), label="Train Loss")
        plt.plot(*zip(*eval_loss), label="Eval Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs. Epoch")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = output_dir / "loss_curve.png"
        plt.savefig(plot_path, dpi=150)
        plt.show()
        print(f"Saved loss curve to: {plot_path}")


if __name__ == "__main__":
    main()
