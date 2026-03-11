from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json
import os
import glob
from sentence_transformers import SentenceTransformer, losses, InputExample, util
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import types
from sentence_transformers.evaluation import InformationRetrievalEvaluator

embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

REPO_PATH = 'WildGraphBench'
DOMAIN = 'culture'
PATH = path = f"{REPO_PATH}/QA/{DOMAIN}/questions.jsonl"

from sklearn.model_selection import train_test_split

# Split questions and gold answers into train/val

def build_training_pairs(questions, all_chunks, top_k=3):
    """
    For each question, retrieve top-k chunks as positives,
    and random distant chunks as negatives.
    """
    examples = []

    # Embed all chunks once
    chunk_embeddings = embed_model.encode(all_chunks, convert_to_tensor=True)

    for q in questions:
        q_emb = embed_model.encode(q, convert_to_tensor=True)

        # Cosine similarity to all chunks
        sims = util.cos_sim(q_emb, chunk_embeddings)[0]

        # Top-k = positives, bottom-k = hard negatives
        top_indices = sims.topk(top_k).indices.tolist()
        bottom_indices = sims.topk(top_k, largest=False).indices.tolist()

        for pos_idx in top_indices:
            for neg_idx in bottom_indices:
                examples.append(InputExample(
                    texts=[q, all_chunks[pos_idx], all_chunks[neg_idx]]
                ))

    return examples

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


questions, gold_answers = load_questions(PATH)
retrieved_positives = get_positive_chunks(questions,
                                          gold_answers,
                                          load_reference_pages("culture",
                                                               'Marvel Cinematic Universe',
                                                               chunk_size=300),
                                          SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'))



# Build pairs — just (question, positive_chunk), no explicit negatives needed
# MNRL uses other items in the batch as negatives


train_questions, val_questions, train_answers, val_answers = train_test_split(
    questions, gold_answers, test_size=0.2, random_state=42
)


all_chunks_with_ids = load_reference_pages("culture", 'Marvel Cinematic Universe', chunk_size=300)

train_positives = get_positive_chunks(train_questions, train_answers, all_chunks_with_ids, embed_model)
val_positives   = get_positive_chunks(val_questions,   val_answers,   all_chunks_with_ids, embed_model)


train_examples = [InputExample(texts=[q, chunk])
                  for q, chunks in zip(train_questions, train_positives)
                  for chunk in chunks]

val_examples = [InputExample(texts=[q, chunk])
                for q, chunks in zip(val_questions, val_positives)
                for chunk in chunks]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
# val_dataloader is used by the evaluator, not fit() directly


# --- TRAIN evaluator ---
train_queries      = {str(i): q for i, q in enumerate(train_questions)}
train_corpus       = {str(i): c for i, c in enumerate([c for chunks in train_positives for c in chunks])}
train_relevant     = {}
idx = 0
for i, chunks in enumerate(train_positives):
    train_relevant[str(i)] = set()
    for _ in chunks:
        train_relevant[str(i)].add(str(idx)); idx += 1

train_evaluator = InformationRetrievalEvaluator(
    queries=train_queries, corpus=train_corpus, relevant_docs=train_relevant,
    name="train-eval", precision_recall_at_k=[1, 3, 5]
)

# --- VAL evaluator (this is what you pass to fit()) ---
val_queries    = {str(i): q for i, q in enumerate(val_questions)}
val_corpus     = {str(i): c for i, c in enumerate([c for chunks in val_positives for c in chunks])}
val_relevant   = {}
idx = 0
for i, chunks in enumerate(val_positives):
    val_relevant[str(i)] = set()
    for _ in chunks:
        val_relevant[str(i)].add(str(idx)); idx += 1

val_evaluator = InformationRetrievalEvaluator(
    queries=val_queries, corpus=val_corpus, relevant_docs=val_relevant,
    name="val-eval", precision_recall_at_k=[1, 3, 5]
)

train_eval_scores = []
val_eval_scores = []



simple_examples = []
for q, chunks in zip(questions, retrieved_positives):
    for chunk in chunks:
        simple_examples.append(InputExample(texts=[q, chunk]))

train_dataloader = DataLoader(simple_examples, shuffle=True, batch_size=16)

# MultipleNegativesRankingLoss is best for retrieval tasks
train_loss = losses.MultipleNegativesRankingLoss(embed_model)

# --- 1. Track loss by wrapping the loss function ---
batch_losses = []
epoch_losses = []
eval_scores = []  # <-- add this

_original_forward_fn = train_loss.__class__.forward

def tracked_forward(self, *args, **kwargs):
    loss = _original_forward_fn(self, *args, **kwargs)
    batch_losses.append(loss.item())
    return loss

train_loss.forward = types.MethodType(tracked_forward, train_loss)

# --- 2. Build an evaluator for accuracy (retrieval precision@k) ---
queries = {str(i): q for i, q in enumerate(questions)}
corpus = {str(i): chunk for i, chunk in enumerate([chunk for chunks in retrieved_positives for chunk in chunks])}


def get_primary_score(result):
    """Extract a single float from an evaluator result (dict or float)."""
    if isinstance(result, dict):
        # Prefer NDCG, otherwise take the first value
        for key in result:
            if 'ndcg' in key.lower():
                return result[key]
        return list(result.values())[0]
    return result  # already a float in older versions

def loss_callback(score, epoch, steps):
    start = int(epoch - 1) * len(train_dataloader)
    end   = int(epoch)     * len(train_dataloader)
    avg   = sum(batch_losses[start:end]) / max(len(batch_losses[start:end]), 1)
    epoch_losses.append(avg)

    val_score   = get_primary_score(score)
    train_score = get_primary_score(train_evaluator(embed_model))

    val_eval_scores.append(val_score)
    train_eval_scores.append(train_score)

    print(f"Epoch {epoch} | Loss: {avg:.4f} | Train: {train_score:.4f} | Val: {val_score:.4f}")


embed_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=50,
    output_path="../mcu-embeddings",
    evaluator=val_evaluator,  # <-- val evaluator drives early stopping & saving
    evaluation_steps=len(train_dataloader),
    callback=loss_callback,
    show_progress_bar=True,
)

epochs_range = range(1, len(epoch_losses) + 1)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_range, epoch_losses, marker='o', color='steelblue')
ax1.set_title("Loss vs Epoch"); ax1.set_xlabel("Epoch"); ax1.grid(True)

ax2.plot(epochs_range, train_eval_scores, marker='o', color='steelblue', label='Train')
ax2.plot(epochs_range, val_eval_scores,   marker='o', color='darkorange', label='Val')
ax2.set_title("Retrieval Score vs Epoch"); ax2.set_xlabel("Epoch")
ax2.legend(); ax2.grid(True)

plt.tight_layout(); plt.show()