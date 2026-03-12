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
from sklearn.model_selection import train_test_split


#===============================================================================
# File paths for getting training and validation data as well as other constants
#===============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
EMBED_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
LOSS_FN = losses.MultipleNegativesRankingLoss(EMBED_MODEL)
EPOCHS = 5
REPO_PATH = "WildGraphBench"
DOMAIN_TUPLES = [('culture', 'Marvel Cinematic Universe'), ('geography', 'United States'),
                 ('health', 'COVID-19 pandemic'), ('history', 'World War II'),
                 ('human_activities', '2022 FIFA World Cup'), ('mathematics', 'Prime number'),
                 ('nature', '2012 Pacific typhoon season'), ('people', 'Donald Trump'),
                 ('philosophy', 'Authoritarian socialism'), ('religion', 'Persecution of Muslims'),
                 ('society', 'Human'), ('technology', 'Steam(service)')]


#==========================
# MODEL DEFINTIONS ETC
#==========================

embed_model = EMBED_MODEL

#========================================
# Function definitions for data retrieval
#========================================


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


def load_reference_pages(domain_tuple, chunk_size=300):
    folder = f"{REPO_PATH}/corpus/{domain_tuple[0]}/{domain_tuple[1]}/reference_pages/"
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


#=================================================
# Function definitions for building training pairs
#=================================================

def get_positive_chunks(questions, gold_answers, all_chunks_with_ids, embed_model, top_k=3, cached_embeddings=None):
    all_chunk_texts = [chunk_tuple[1] for chunk_tuple in all_chunks_with_ids]

    # Use cached embeddings if provided, otherwise compute
    chunk_embeddings = cached_embeddings if cached_embeddings is not None \
        else embed_model.encode(all_chunk_texts, convert_to_tensor=True)

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


def build_training_pairs(questions, all_chunks, top_k=3):
    """
    For each question, retrieve top-k chunks as positives,
    and random distant chunks as negatives.
    """
    examples = []
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


#=========================================
#   Start constructing training pairs
#=========================================


references = []
questions = []
gold_answers = []

for tuple in DOMAIN_TUPLES:
    references.extend(load_reference_pages(tuple, chunk_size=300))
    temp_questions, temp_gold_answers = load_questions(f"{REPO_PATH}/QA/{tuple[0]}/questions.jsonl")
    questions.extend(temp_questions)
    gold_answers.extend(temp_gold_answers)

# Split FIRST, then encode chunks only once
train_questions, val_questions, train_answers, val_answers = train_test_split(
    questions, gold_answers, test_size=0.2, random_state=42
)

# Encode chunks once, reuse for both train and val
all_chunk_texts = [chunk[1] for chunk in references]
print("Encoding corpus chunks (once)...")
cached_chunk_embeddings = embed_model.encode(all_chunk_texts, convert_to_tensor=True, show_progress_bar=True)

train_positives = get_positive_chunks(train_questions, train_answers, references, embed_model, cached_embeddings=cached_chunk_embeddings)
val_positives   = get_positive_chunks(val_questions,   val_answers,   references, embed_model, cached_embeddings=cached_chunk_embeddings)


train_examples = [InputExample(texts=[q, chunk])
                  for q, chunks in zip(train_questions, train_positives)
                  for chunk in chunks]

val_examples = [InputExample(texts=[q, chunk])
                for q, chunks in zip(val_questions, val_positives)
                for chunk in chunks]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

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

# MultipleNegativesRankingLoss is best for retrieval tasks
# Maybe try different loss functions to see how performance drop off is?
train_loss = LOSS_FN

batch_losses = []
epoch_losses = []
eval_scores = []

# Track loss by wrapping the loss function
_original_forward_fn = train_loss.__class__.forward

#===========================================================================
# Function Definitions for calculating per epoch performance (score & loss)
#===========================================================================

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
    val_score = get_primary_score(score)
    val_eval_scores.append(val_score)
    # Only compute expensive train eval every 2 epochs
    if epoch % 2 == 0 or epoch == EPOCHS:
        train_score = get_primary_score(train_evaluator(embed_model))
        train_eval_scores.append((epoch, train_score))  # store epoch too since we skip some
        print(f"Epoch {epoch} | Loss: {avg:.4f} | Train: {train_score:.4f} | Val: {val_score:.4f}")
    else:
        print(f"Epoch {epoch} | Loss: {avg:.4f} | Val: {val_score:.4f}")

#=========================================================
#                    Train the model
#=========================================================


embed_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=50,
    output_path="../all_embeddings",
    evaluator=val_evaluator,  # <-- val evaluator drives early stopping & saving
    evaluation_steps=len(train_dataloader),
    callback=loss_callback,
    show_progress_bar=True,
)


#========================================================
#                     Plot Results
#========================================================

epochs_range = range(1, len(epoch_losses) + 1)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_range, epoch_losses, marker='o', color='steelblue')
ax1.set_title("Loss vs Epoch"); ax1.set_xlabel("Epoch"); ax1.grid(True)
train_epochs = [e for e, _ in train_eval_scores]
train_scores = [s for _, s in train_eval_scores]
ax2.plot(epochs_range, val_eval_scores, marker='o', color='darkorange', label='Val')
ax2.plot(train_epochs, train_scores,    marker='o', color='steelblue',  label='Train')
ax2.set_title("Retrieval Score vs Epoch")
ax2.set_xlabel("Epoch")
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()
