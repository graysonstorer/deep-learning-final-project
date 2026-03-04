# search/ranker.py
# The master scoring pipeline.
# Combines GLiNER2, embedding similarity, GraphRAG, and Llama into a final ranking.

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class Ranker:
    """
    Combines all scoring signals into a final ranking of link candidates.

    Pipeline for each hop:
      1. GLiNER filter: reduce 200+ links → TOP_K candidates
      2. Embedding similarity: score remaining candidates vs target
      3. GraphRAG re-rank: apply community/proximity bonuses
      4. Llama reorder: LLM picks final top candidate with reasoning
      5. Merge: weighted combination → final sorted list
    """

    def __init__(self, embedder, gliner_filter, graph_rag, llama_agent=None):
        self.embedder = embedder
        self.gliner = gliner_filter
        self.graph_rag = graph_rag
        self.llama = llama_agent  # may be None if --no-llm

        self._target_embedding: Optional[np.ndarray] = None
        self._target_title: str = ""

    def set_target(self, title: str, lede: str):
        """Prepare target embedding. Call once at the start."""
        self._target_title = title
        target_text = self.embedder.build_target_text(title, lede)
        self._target_embedding = self.embedder.embed(target_text)
        self.gliner.index_target(title, lede)
        self.graph_rag.set_target(title, self._target_embedding)
        logger.info(f"Ranker target set: '{title}'")

    def rank(
        self,
        current_page_title: str,
        current_page_lede: str,
        candidate_links: list[str],
        visited_pages: set[str],
        verbose: bool = False,
    ) -> list[tuple[str, float, str]]:
        """
        Rank all candidate links from the current page.

        Returns list of (title, final_score, reasoning) sorted desc.
        """
        import config

        if not candidate_links:
            return []

        # ── Stage 1: GLiNER filter ───────────────────────────────────────────
        gliner_scored = self.gliner.filter_and_score(
            candidates=candidate_links,
            max_results=config.MAX_LINKS_AFTER_GLINER,
        )
        filtered_titles = [t for t, _ in gliner_scored]
        gliner_score_map = {t: s for t, s in gliner_scored}

        if verbose:
            logger.debug(f"GLiNER filter: {len(candidate_links)} → {len(filtered_titles)}")

        # ── Stage 2: Embedding similarity ───────────────────────────────────
        # Build (title, text_to_embed) pairs
        embed_candidates = [
            (title, self.embedder.build_candidate_text(title))
            for title in filtered_titles
        ]
        embed_scored = self.embedder.score_candidates(
            self._target_embedding, embed_candidates
        )
        embed_score_map = {t: s for t, s in embed_scored}

        # ── Stage 3: GraphRAG re-rank ────────────────────────────────────────
        graphrag_score_map = {}
        for title in filtered_titles:
            graphrag_score_map[title] = self.graph_rag.score_candidate(title)

        # ── Stage 4: Cycle penalty ────────────────────────────────────────────
        def cycle_penalty(title: str) -> float:
            if title in visited_pages:
                return config.CYCLE_PENALTY
            return 1.0

        # ── Stage 5: Weighted combination (pre-LLM) ──────────────────────────
        combined_scores = {}
        for title in filtered_titles:
            score = (
                config.WEIGHT_EMBEDDING * embed_score_map.get(title, 0)
                + config.WEIGHT_GLINER * gliner_score_map.get(title, 0)
                + config.WEIGHT_GRAPHRAG * graphrag_score_map.get(title, 0)
            )
            score *= cycle_penalty(title)
            combined_scores[title] = score

        # Sort by combined score
        pre_llm_ranked = sorted(
            filtered_titles,
            key=lambda t: combined_scores[t],
            reverse=True
        )

        # ── Stage 6: Llama reranking (optional) ──────────────────────────────
        reasoning_map = {}
        if self.llama and config.USE_LLM and len(pre_llm_ranked) > 0:
            top_k = pre_llm_ranked[:config.TOP_K_FOR_LLM]
            top_k_with_scores = [(t, combined_scores[t]) for t in top_k]

            try:
                llm_choice, reasoning = self.llama.pick_best_hop(
                    current_page=current_page_title,
                    target_page=self._target_title,
                    candidates=top_k_with_scores,
                )
                reasoning_map[llm_choice] = reasoning

                # Boost the LLM's chosen page
                if llm_choice in combined_scores:
                    combined_scores[llm_choice] += config.WEIGHT_LLM
                    logger.info(f"LLM chose: '{llm_choice}' — {reasoning[:100]}")

            except Exception as e:
                logger.warning(f"LLM ranking failed, using embedding ranking: {e}")

        # ── Final sort ────────────────────────────────────────────────────────
        final_ranked = sorted(
            filtered_titles,
            key=lambda t: combined_scores[t],
            reverse=True
        )

        if verbose or logger.isEnabledFor(logging.DEBUG):
            logger.debug("Top 5 candidates:")
            for t in final_ranked[:5]:
                logger.debug(
                    f"  {t:40s} embed={embed_score_map.get(t, 0):.3f} "
                    f"gliner={gliner_score_map.get(t, 0):.3f} "
                    f"graph={graphrag_score_map.get(t, 0):.3f} "
                    f"final={combined_scores.get(t, 0):.3f}"
                )

        return [
            (title, combined_scores[title], reasoning_map.get(title, ""))
            for title in final_ranked
        ]
