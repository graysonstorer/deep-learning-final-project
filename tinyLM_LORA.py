import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import networkx as nx
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
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


# Load model once outside the function (so it doesn't reload on every call)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,   # use float32 if on CPU
#     device_map=torch.device('mps'),           # automatically uses GPU if available
# )



# embedder = SentenceTransformer("all-MiniLM-L6-v2")

REPO_PATH = 'WildGraphBench'
DOMAIN = 'culture'
PATH = f"{REPO_PATH}/QA/{DOMAIN}/questions.jsonl"
TOPIC = 'Marvel Cinematic Universe'

def make_finetune_dataset(questions, retrieved_chunks_list, answers):
    """Build prompt-completion pairs from your QA data."""
    records = []
    for q, chunks, a in zip(questions, retrieved_chunks_list, answers):
        context = "\n\n".join(chunks)
        prompt = f"""<|system|>
You are a helpful assistant answering questions about the Marvel Cinematic Universe.
</s>
<|user|>
Context:
{context}

Question: {q}
</s>
<|assistant|>
{a}</s>"""
        records.append({"text": prompt})
    return Dataset.from_list(records)


import glob
def load_questions(path):
    questions, gold_answers = [], []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if "answer" not in item:
                continue  # skip malformed entries
            questions.append(item["question"])
            gold_answers.append(item["answer"])
    return questions, gold_answers




def get_positive_chunks(questions, gold_answers, all_chunks_with_ids, embed_model, top_k=3):
    """
    For each question, find the chunks most relevant to its gold answer.
    Returns a list of lists: retrieved_positives[i] = [chunk_text1, chunk_text2, ...] for questions[i]
    """
    from sentence_transformers import util

    all_chunk_texts = [chunk_tuple[1] for chunk_tuple in all_chunks_with_ids]
    chunk_embeddings = embed_model.encode(all_chunk_texts, convert_to_tensor=True)
    retrieved_positives = []

    for question, answer in zip(questions, gold_answers):
        # First try: find chunks that literally contain part of the answer
        answer_words = set(answer.lower().split())
        scored = []
        for chunk_text in all_chunk_texts:
            chunk_words = set(chunk_text.lower().split())
            overlap = len(answer_words & chunk_words) / max(len(answer_words), 1)
            scored.append(overlap)

        best_overlap = max(scored)

        if best_overlap > 0.3:
            # Use word overlap to find positives
            top_indices = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)[:top_k]
        else:
            # Fallback: embed the answer and find similar chunks
            answer_emb = embed_model.encode(answer, convert_to_tensor=True)
            sims = util.cos_sim(answer_emb, chunk_embeddings)[0]
            top_indices = sims.topk(top_k).indices.tolist()

        retrieved_positives.append([all_chunk_texts[i] for i in top_indices])

    return retrieved_positives


def load_reference_pages(domain, topic, chunk_size=300):
    folder = f"{REPO_PATH}/corpus/{domain}/{topic}/reference_pages/"
    all_chunks = []
    for filepath in glob.glob(folder + "*.txt"):
        with open(filepath, "r", errors="ignore") as f:
            text = f.read()
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            node_id = f"{os.path.basename(filepath)}__{i}"
            all_chunks.append((node_id, chunk))
    return all_chunks


from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments
from trl import SFTTrainer

lora_config = LoraConfig(
    r=16,                        # rank — higher = more capacity, more memory
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # attention layers only
    lora_dropout=0.45,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,   # use float32 if on CPU
    device_map=torch.device('cpu'),           # automatically uses GPU if available
), lora_config)
model.print_trainable_parameters()



questions, gold_answers = load_questions(PATH)
all_chunks = load_reference_pages(DOMAIN, TOPIC)
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
retrieved_positives = get_positive_chunks(questions, gold_answers, all_chunks, embed_model)


def make_lora_dataset(questions, gold_answers, retrieved_positives):
    """
    Each training example = the context chunks + question → gold answer,
    formatted in TinyLlama's chat template.
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

dataset = make_lora_dataset(questions, gold_answers, retrieved_positives)
print(f"Dataset size: {len(dataset)} examples")
print("\nSample:")
print(dataset[0]["text"])



dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

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

    # Add these:
    eval_strategy="epoch",   # evaluate at the end of every epoch
    per_device_eval_batch_size=2,
    load_best_model_at_end=False,   # optional but useful
)

def formatting_func(example):
    return example["text"]

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    formatting_func=formatting_func
    # max_seq_length=2048
)
trainer.train()


import matplotlib.pyplot as plt

log_history = trainer.state.log_history

train_loss = [(e["epoch"], e["loss"]) for e in log_history if "loss" in e]
eval_loss  = [(e["epoch"], e["eval_loss"]) for e in log_history if "eval_loss" in e]

plt.plot(*zip(*train_loss), label="Train Loss")
plt.plot(*zip(*eval_loss),  label="Eval Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs. Epoch")
plt.legend()
plt.grid(True)
plt.show()