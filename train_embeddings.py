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

relevant_docs = {}
idx = 0
for i, pos_chunks in enumerate(retrieved_positives):
    relevant_docs[str(i)] = set()
    for chunk in pos_chunks:
        relevant_docs[str(i)].add(str(idx))
        idx += 1

evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="mcu-eval",
    precision_recall_at_k=[1, 3, 5],
)

# --- 3. Callback to record epoch losses and eval scores ---
steps_per_epoch = len(train_dataloader)

def loss_callback(score, epoch, steps):
    start = int(epoch - 1) * steps_per_epoch
    end = int(epoch) * steps_per_epoch
    avg = sum(batch_losses[start:end]) / max(len(batch_losses[start:end]), 1)
    epoch_losses.append(avg)
    eval_scores.append(score)  # <-- collect directly from callback
    print(f"Epoch {epoch} | Avg Loss: {avg:.4f} | Eval Score: {score:.4f}")

# --- 4. Train ---
embed_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=20,
    warmup_steps=50,
    output_path="../mcu-embeddings",
    evaluator=evaluator,
    evaluation_steps=steps_per_epoch,
    callback=loss_callback,
    show_progress_bar=True,
)

# --- 5. Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

epochs_range = range(1, len(epoch_losses) + 1)

ax1.plot(epochs_range, epoch_losses, marker='o', color='steelblue')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Avg Loss")
ax1.set_title("Loss vs Epoch")
ax1.grid(True)

ax2.plot(range(1, len(eval_scores) + 1), eval_scores, marker='o', color='darkorange')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Eval Score")
ax2.set_title("Retrieval Score vs Epoch")
ax2.grid(True)

plt.tight_layout()
plt.show()

