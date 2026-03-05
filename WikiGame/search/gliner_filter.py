# search/gliner_filter.py
# GLiNER2: Zero-shot named entity recognition for link candidate pruning.
#
# Workflow:
#   1. Extract entities from the TARGET page (done once at start)
#   2. For each hop, score link candidates by how well their titles/text
#      match the target's entities
#   3. Return a filtered + boosted candidate list

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import
_gliner_model = None


def _get_gliner():
    global _gliner_model
    if _gliner_model is None:
        from gliner import GLiNER
        import config
        logger.info(f"Loading GLiNER model: {config.GLINER_MODEL}")
        _gliner_model = GLiNER.from_pretrained(config.GLINER_MODEL)
        logger.info("GLiNER model loaded.")
    return _gliner_model


class GlinerFilter:
    """
    Uses GLiNER2 to extract named entities from the target page,
    then scores and filters link candidates based on entity overlap.

    This is the first-pass filter: reduces 200+ links → ~25 candidates
    before embedding and LLM stages.
    """

    def __init__(self):
        self._target_entities: list[dict] = []   # [{text, label, score}, ...]
        self._target_entity_texts: set[str] = set()
        self._model = None

    def _model(self):
        return _get_gliner()

    def extract_entities(self, text: str) -> list[dict]:
        """
        Extract named entities from a text string.
        Returns list of {text, label, score} dicts.
        """
        import config
        try:
            model = _get_gliner()
            entities = model.predict_entities(
                text[:2000],  # GLiNER works best on shorter inputs
                config.GLINER_ENTITY_TYPES,
                threshold=config.GLINER_THRESHOLD,
            )
            return entities
        except Exception as e:
            logger.warning(f"GLiNER extraction failed: {e}")
            return []

    def index_target(self, target_title: str, target_lede: str):
        """
        Extract entities from the target page and cache them.
        Call this once at the start of a game.
        """
        text = f"{target_title}. {target_lede}"
        self._target_entities = self.extract_entities(text)

        # Also always include the target title itself and its words
        self._target_entity_texts = set()
        self._target_entity_texts.add(target_title.lower())
        for word in target_title.split():
            if len(word) > 3:
                self._target_entity_texts.add(word.lower())

        for ent in self._target_entities:
            self._target_entity_texts.add(ent["text"].lower())
            # Add individual words from multi-word entities
            for word in ent["text"].split():
                if len(word) > 3:
                    self._target_entity_texts.add(word.lower())

        logger.info(f"Target entities indexed: {[e['text'] for e in self._target_entities[:10]]}")

    def score_candidate(self, title: str, lede: str = "") -> float:
        """
        Score a single candidate link based on entity overlap with target.
        Returns a score in [0, 1].
        """
        if not self._target_entity_texts:
            return 0.0

        text = f"{title} {lede}".lower()
        score = 0.0

        # Check for entity matches
        matches = 0
        for entity_text in self._target_entity_texts:
            if entity_text in text:
                matches += 1

        if matches > 0:
            # Normalize: more matches → higher score, but diminishing returns
            score = min(matches / max(len(self._target_entity_texts), 1), 1.0)
            # Extra bonus for exact title match
            if title.lower() in self._target_entity_texts:
                score = min(score + 0.3, 1.0)

        return score

    def filter_and_score(
        self,
        candidates: list[str],
        candidate_lede_map: dict[str, str] = None,
        max_results: int = None,
    ) -> list[tuple[str, float]]:
        """
        Score and filter a list of candidate link titles.

        candidates: list of Wikipedia page titles
        candidate_lede_map: optional {title: lede_text} for richer scoring
        max_results: cap the output (uses config.MAX_LINKS_AFTER_GLINER if None)

        Returns list of (title, gliner_score) sorted desc.
        """
        import config
        if max_results is None:
            max_results = config.MAX_LINKS_AFTER_GLINER

        lede_map = candidate_lede_map or {}

        scored = []
        for title in candidates:
            lede = lede_map.get(title, "")
            score = self.score_candidate(title, lede)
            scored.append((title, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Strategy: always keep top-max_results, but prioritize entity matches
        # If we have many high-scoring items, great. If not, backfill with unscored ones.
        high_score = [s for s in scored if s[1] > 0]
        zero_score = [s for s in scored if s[1] == 0]

        result = high_score[:max_results]
        if len(result) < max_results:
            # Backfill with non-matching candidates (alphabetical for diversity)
            backfill_needed = max_results - len(result)
            result += zero_score[:backfill_needed]

        return result[:max_results]

    def get_target_entities(self) -> list[str]:
        """Return the list of extracted target entity strings."""
        return [e["text"] for e in self._target_entities]
