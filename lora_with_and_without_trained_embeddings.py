import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
import accelerate
import os, json, glob

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# =============================
#       Various Constants
# =============================

REPO_PATH = 'WildGraphBench'
DOMAIN = 'culture'
PATH = f"{REPO_PATH}/QA/{DOMAIN}/questions.jsonl"
TOPIC = 'Marvel Cinematic Universe'
LIMIT = 50 # <-- number of questions to run through for evaluation.
WORD_COVERAGE_THRESHOLD = 0.5  # <-- hyper param for when calculating retrieval recall
# (how much does a word need to overlap for it to be considered overlapping)

# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────

def load_questions(path):
    questions, gold_answers = [], []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if "answer" not in item:
                continue
            questions.append(item["question"])
            gold_answers.append(item["answer"])
    return questions, gold_answers


def load_reference_pages(domain, topic, chunk_size=300):
    folder = f"{REPO_PATH}/corpus/{domain}/{topic}/reference_pages/"
    all_chunks = []
    for filepath in glob.glob(folder + "*.txt"):
        with open(filepath, "r", errors="ignore") as f:
            text = f.read()
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            node_id = f"{os.path.basename(filepath)}__{i}"
            all_chunks.append((node_id, chunk))
    return all_chunks


def get_positive_chunks(questions, gold_answers, all_chunks_with_ids, embed_model, top_k=3):
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


# =============================================
# Evaluation helpers
# =============================================

def retrieval_recall_at_k(questions, gold_answers, all_chunk_texts, embed_model,
                          k=3, word_coverage_threshold=0.5):
    """
    Recall@k with two complementary checks — a question is a hit if EITHER:

    1. Exact-phrase hit: the full gold answer string (lowercased) appears
       as a substring in the concatenated retrieved chunks. Catches short,
       specific answers (names, dates, titles) precisely.

    2. Word-coverage hit: the fraction of meaningful answer words (>3 chars,
       non-stopword) found in retrieved chunks meets `word_coverage_threshold`.
       Defaults to 0.5 — at least half the content words must appear.

    The original any(w in chunks) bug caused saturation at 1.0 because a
    single common word trivially matched everywhere.
    """
    STOPWORDS = {
        "that", "with", "this", "have", "from", "they", "been", "were",
        "their", "there", "when", "which", "will", "would", "could",
        "should", "about", "into", "than", "then", "also", "some",
    }

    chunk_embeddings = embed_model.encode(
        all_chunk_texts, convert_to_tensor=True, show_progress_bar=False
    )
    hits = 0

    for question, answer in zip(questions, gold_answers):
        q_emb = embed_model.encode(question, convert_to_tensor=True)
        sims = util.cos_sim(q_emb, chunk_embeddings)[0]
        top_indices = sims.topk(k).indices.tolist()
        top_chunks = " ".join(all_chunk_texts[i] for i in top_indices).lower()
        answer_lower = answer.lower().strip()
        exact_hit = answer_lower in top_chunks
        answer_words = [
            w for w in answer_lower.split()
            if len(w) > 3 and w not in STOPWORDS
        ]
        if answer_words:
            coverage = sum(1 for w in answer_words if w in top_chunks) / len(answer_words)
            coverage_hit = coverage >= word_coverage_threshold
        else:
            coverage_hit = False  # no meaningful words — don't count as hit
        if exact_hit or coverage_hit:
            hits += 1
    return hits / max(len(questions), 1)



def answer_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer."""
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


def generate_answer(model, tokenizer, question, context, max_new_tokens=100):
    """Run greedy generation for a single QA example."""
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
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def evaluate_rag(model, tokenizer, questions, gold_answers,
                 all_chunk_texts, embed_model, label="", k=3, n_gen=20):
    """
    Full RAG evaluation:
      - Retrieval Recall@k
      - Generation F1 on a subset of n_gen questions
    Returns a dict of metrics.
    """
    recall = retrieval_recall_at_k(questions, gold_answers, all_chunk_texts, embed_model, k=k)

    # Generation F1 (subset for speed)
    subset = list(zip(questions, gold_answers))[:n_gen]
    chunk_embeddings = embed_model.encode(all_chunk_texts, convert_to_tensor=True, show_progress_bar=False)
    f1_scores = []
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


# =============================================
# Callback: evaluate both embedders each epoch
# =============================================

class EmbeddingEvalCallback(TrainerCallback):
    """
    At the end of every epoch, evaluates RAG quality using
    (a) the base embedder and (b) the fine-tuned embedder,
    then stores results for later plotting.
    """
    def __init__(self, tokenizer, questions, gold_answers, all_chunk_texts,
                 base_embedder, finetuned_embedder, eval_every_n_epochs=1,
                 k=3, n_gen=20):
        self.tokenizer = tokenizer
        self.questions = questions
        self.gold_answers = gold_answers
        self.all_chunk_texts = all_chunk_texts
        self.base_embedder = base_embedder
        self.finetuned_embedder = finetuned_embedder
        self.eval_every_n_epochs = eval_every_n_epochs
        self.k = k
        self.n_gen = n_gen

        # Storage: {epoch: {"base": {...}, "finetuned": {...}}}
        self.results: dict = {}

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch))
        if epoch % self.eval_every_n_epochs != 0:
            return

        print(f"\n── Embedding Eval @ Epoch {epoch} ──")
        model.eval()

        base_metrics = evaluate_rag(
            model, self.tokenizer,
            self.questions, self.gold_answers,
            self.all_chunk_texts, self.base_embedder,
            label="base", k=self.k, n_gen=self.n_gen,
        )
        ft_metrics = evaluate_rag(
            model, self.tokenizer,
            self.questions, self.gold_answers,
            self.all_chunk_texts, self.finetuned_embedder,
            label="finetuned", k=self.k, n_gen=self.n_gen,
        )

        self.results[epoch] = {"base": base_metrics, "finetuned": ft_metrics}
        model.train()

    def plot(self, save_path="embedding_comparison.png"):
        if not self.results:
            print("No evaluation results to plot.")
            return
        epochs = sorted(self.results.keys())
        base_recall  = [self.results[e]["base"]["recall"]   for e in epochs]
        ft_recall    = [self.results[e]["finetuned"]["recall"] for e in epochs]
        base_f1      = [self.results[e]["base"]["gen_f1"]   for e in epochs]
        ft_f1        = [self.results[e]["finetuned"]["gen_f1"] for e in epochs]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(epochs, base_recall, "o-", label="Base embedder", color="steelblue")
        axes[0].plot(epochs, ft_recall,   "s--", label="Fine-tuned embedder", color="darkorange")
        axes[0].set_title("Retrieval Recall@k vs Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Recall@k")
        axes[0].legend()
        axes[0].grid(True)
        axes[1].plot(epochs, base_f1, "o-", label="Base embedder", color="steelblue")
        axes[1].plot(epochs, ft_f1,  "s--", label="Fine-tuned embedder", color="darkorange")
        axes[1].set_title("Generation F1 vs Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Token F1")
        axes[1].legend()
        axes[1].grid(True)
        plt.suptitle("Base vs Fine-tuned Embeddings — RAG Performance During LoRA Training")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"Plot saved to {save_path}")


# =============================================
# Main training setup
# =============================================

from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments
from trl import SFTTrainer

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.45,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(
    AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map=torch.device('cpu'),
    ),
    lora_config,
)
model.print_trainable_parameters()

# Load data
questions, gold_answers = load_questions(PATH)
all_chunks = load_reference_pages(DOMAIN, TOPIC)
all_chunk_texts = [c[1] for c in all_chunks]

# Both embedders
base_embedder     = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # <-- run-of-the-mill ST
finetuned_embedder = SentenceTransformer("./all_embeddings") # <-- embedder trained on everything in WildGraph

# Build training dataset using base embedder for retrieval
retrieved_positives = get_positive_chunks(questions, gold_answers, all_chunks, base_embedder)
dataset = make_lora_dataset(questions, gold_answers, retrieved_positives)
print(f"Dataset size: {len(dataset)} examples")
print("\nSample:")
print(dataset[0]["text"])

dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset  = dataset["test"]
print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# ── Instantiate the callback ──
embedding_eval_cb = EmbeddingEvalCallback(
    tokenizer=tokenizer,
    questions=questions[:LIMIT],          # limit to 50 for speed; remove cap for full eval
    gold_answers=gold_answers[:LIMIT],
    all_chunk_texts=all_chunk_texts,
    base_embedder=base_embedder,
    finetuned_embedder=finetuned_embedder,
    eval_every_n_epochs=1,             # evaluate every epoch
    k=3,
    n_gen=20,                          # generate answers for first 20 questions
)

training_args = TrainingArguments(
    output_dir="tinyllama-lora-mcu",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    use_mps_device=False,
    use_cpu=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    per_device_eval_batch_size=2,
    load_best_model_at_end=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    formatting_func=lambda example: example["text"],
    callbacks=[embedding_eval_cb],     # ← attach the callback here
)

trainer.train()

# ── Plot train/eval loss ──
log_history = trainer.state.log_history
train_loss = [(e["epoch"], e["loss"])      for e in log_history if "loss"      in e]
eval_loss  = [(e["epoch"], e["eval_loss"]) for e in log_history if "eval_loss" in e]

plt.figure(figsize=(7, 4))
plt.plot(*zip(*train_loss), label="Train Loss")
plt.plot(*zip(*eval_loss),  label="Eval Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Loss vs. Epoch"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig("loss_curve.png", dpi=150); plt.show()

# ── Plot embedding comparison ──
embedding_eval_cb.plot("embedding_comparison.png")

# ── Dump raw numbers ──
print("\n── Full Embedding Eval Results ──")
for epoch, metrics in sorted(embedding_eval_cb.results.items()):
    print(f"Epoch {epoch}: {metrics}")