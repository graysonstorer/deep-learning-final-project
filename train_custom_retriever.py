from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from custom_retriever_bridge import (
    CUSTOM_LINKS_PATH,
    CUSTOM_PAGES_PATH,
    build_custom_supervision_pairs,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_ir_evaluator(
    pairs: Sequence[Tuple[str, str, int]],
    name: str,
) -> InformationRetrievalEvaluator:
    queries: Dict[str, str] = {}
    corpus: Dict[str, str] = {}
    relevant_docs: Dict[str, set[str]] = {}

    for i, (query_text, doc_text, target_page_id) in enumerate(pairs):
        qid = str(i)
        did = str(target_page_id)
        queries[qid] = query_text
        corpus[did] = doc_text
        relevant_docs[qid] = {did}

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        precision_recall_at_k=[1, 3, 5],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a SentenceTransformer retriever on the custom dataset.")
    parser.add_argument("--pages_path", type=str, default=str(CUSTOM_PAGES_PATH))
    parser.add_argument("--links_path", type=str, default=str(CUSTOM_LINKS_PATH))
    parser.add_argument("--output_path", type=str, default="custom_all_embeddings")
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base SentenceTransformer checkpoint",
    )
    parser.add_argument("--max_pages", type=int, default=None)
    parser.add_argument("--max_edges", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--eval_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    args = parser.parse_args()

    set_seed(int(args.seed))

    model = SentenceTransformer(args.model_name, device=args.device)

    pairs = build_custom_supervision_pairs(
        max_pages=args.max_pages,
        max_edges=args.max_edges,
        pages_path=Path(args.pages_path),
        links_path=Path(args.links_path),
    )
    if len(pairs) < 10:
        raise ValueError(
            f"Not enough supervision pairs ({len(pairs)}) built from the custom dataset. "
            "Check pages/links paths or raise --max_edges."
        )

    train_pairs, val_pairs = train_test_split(
        pairs,
        test_size=args.eval_fraction,
        random_state=args.seed,
        shuffle=True,
    )

    print(f"[custom-retriever] supervision pairs: {len(pairs)}")
    print(f"[custom-retriever] train pairs: {len(train_pairs)}")
    print(f"[custom-retriever] val pairs: {len(val_pairs)}")

    train_examples = [InputExample(texts=[query, doc]) for query, doc, _ in train_pairs]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    evaluator = build_ir_evaluator(val_pairs, name="custom-retriever-val")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        evaluator=evaluator,
        evaluation_steps=max(len(train_dataloader), 1),
        output_path=str(output_path),
        show_progress_bar=True,
        use_amp=torch.cuda.is_available(),
    )

    model.save(str(output_path))
    print(f"[custom-retriever] saved checkpoint: {output_path}")


if __name__ == "__main__":
    main()

