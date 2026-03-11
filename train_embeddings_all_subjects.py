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
EMBED_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
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

def get_positive_chunks(questions, gold_answers, all_chunks_with_ids, embed_model, top_k=3):
    """
    For each question, find the chunks most relevant to its gold answer.
    Returns a list of lists: retrieved_positives[i] = [chunk_text1, chunk_text2, ...] for questions[i]
    """
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
    temp_questions, temp_gold_answers = load_questions(path = f"{REPO_PATH}/QA/{tuple[0]}/questions.jsonl")
    questions.extend(temp_questions)
    gold_answers.extend(temp_gold_answers)










