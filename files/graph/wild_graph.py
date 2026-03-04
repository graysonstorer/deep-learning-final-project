# graph/wild_graph.py
# WildGraph: Traversal engine for the Wikipedia game.
# Supports greedy (single path) and beam search (multiple parallel paths).
# Handles cycle detection, path recording, and frontier management.

import heapq
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Path:
    """A single candidate path through Wikipedia pages."""
    pages: list[str]           # pages visited in order
    scores: list[float]        # score of each hop decision
    total_score: float = 0.0   # cumulative score (higher = better)

    def current_page(self) -> str:
        return self.pages[-1]

    def length(self) -> int:
        return len(self.pages) - 1  # number of hops taken

    def has_visited(self, title: str) -> bool:
        return title in self.pages

    def extend(self, next_page: str, score: float) -> "Path":
        """Return a new Path with next_page appended."""
        return Path(
            pages=self.pages + [next_page],
            scores=self.scores + [score],
            total_score=self.total_score + score,
        )

    def average_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def __lt__(self, other: "Path"):
        # For heapq (min-heap), we negate so highest score is popped first
        return self.total_score > other.total_score

    def __repr__(self):
        return f"Path({' → '.join(self.pages)}, score={self.total_score:.3f})"


@dataclass
class TraversalResult:
    """Result of a complete traversal attempt."""
    success: bool
    path: Optional[Path]
    hops_taken: int
    pages_explored: int
    failure_reason: str = ""

    def summary(self) -> str:
        if self.success and self.path:
            route = " → ".join(self.path.pages)
            return (f"✅ Reached target in {self.hops_taken} hops!\n"
                    f"Path: {route}\n"
                    f"Average hop score: {self.path.average_score():.3f}")
        else:
            return f"❌ Failed: {self.failure_reason} (after {self.hops_taken} hops)"


class WildGraph:
    """
    Traversal engine implementing greedy and beam search strategies.

    Usage:
        wg = WildGraph(strategy="beam", beam_width=3)
        wg.initialize(start_page="Pizza", target_page="Alan Turing")

        while not wg.is_done():
            # Get the current page(s) to evaluate
            current_pages = wg.get_frontier()

            for page in current_pages:
                # Your scoring pipeline produces ranked candidates
                ranked = scorer.rank(page, target)
                wg.advance(page, ranked)

        result = wg.get_result()
    """

    def __init__(self, strategy: str = "beam", beam_width: int = 3, max_hops: int = 30):
        self.strategy = strategy
        self.beam_width = beam_width
        self.max_hops = max_hops

        self._active_paths: list[Path] = []
        self._completed_path: Optional[Path] = None
        self._target: str = ""
        self._start: str = ""
        self._all_visited: set[str] = set()    # across ALL paths (for stats)
        self._done: bool = False
        self._failure_reason: str = ""
        self._hop_log: list[dict] = []          # full hop-by-hop log

    def initialize(self, start_page: str, target_page: str):
        """Set up a new traversal."""
        self._start = start_page
        self._target = target_page
        self._active_paths = [Path(pages=[start_page], scores=[])]
        self._all_visited = {start_page}
        self._done = (start_page.lower() == target_page.lower())
        if self._done:
            self._completed_path = self._active_paths[0]
        logger.info(f"WildGraph initialized: '{start_page}' → '{target_page}'")

    def is_done(self) -> bool:
        return self._done

    def get_frontier(self) -> list[str]:
        """
        Return the current page(s) to evaluate next.
        In greedy mode: [best_path.current_page()]
        In beam mode: [path.current_page() for path in top-K active paths]
        """
        if self._done or not self._active_paths:
            return []

        if self.strategy == "greedy":
            return [self._active_paths[0].current_page()]
        else:  # beam
            return [p.current_page() for p in self._active_paths]

    def get_current_path(self) -> Optional[Path]:
        """Return the best active path (for display purposes)."""
        if self._active_paths:
            return self._active_paths[0]
        return None

    def advance(
        self,
        from_page: str,
        ranked_candidates: list[tuple[str, float]],
        reasoning: str = "",
    ):
        """
        Advance the traversal from from_page by picking the best candidate.

        ranked_candidates: list of (page_title, score) sorted desc by score
        reasoning: optional string explanation (from Llama) for logging
        """
        if self._done:
            return

        # Find the path(s) that are currently at from_page
        paths_from_this_page = [p for p in self._active_paths
                                 if p.current_page() == from_page]

        new_paths = []
        for path in paths_from_this_page:
            for candidate_title, score in ranked_candidates:
                # Apply cycle penalty for already-visited pages
                import config
                if path.has_visited(candidate_title):
                    score *= config.CYCLE_PENALTY

                new_path = path.extend(candidate_title, score)
                new_paths.append(new_path)

                # Log this hop
                self._log_hop(
                    from_page=from_page,
                    to_page=candidate_title,
                    score=score,
                    reasoning=reasoning,
                    path_length=new_path.length(),
                )
                self._all_visited.add(candidate_title)

                # Check if we reached the target
                if candidate_title.lower() == self._target.lower():
                    self._completed_path = new_path
                    self._done = True
                    logger.info(f"🎯 Target reached! Path: {new_path}")
                    return

        # Keep only the best paths that didn't originate from this page
        other_paths = [p for p in self._active_paths
                       if p.current_page() != from_page]

        # Merge and keep top beam_width paths
        all_paths = other_paths + new_paths
        all_paths.sort(key=lambda p: p.total_score, reverse=True)

        if self.strategy == "greedy":
            self._active_paths = all_paths[:1]
        else:
            self._active_paths = all_paths[:self.beam_width]

        # Check max hops
        if self._active_paths and self._active_paths[0].length() >= self.max_hops:
            self._done = True
            self._failure_reason = f"Max hops ({self.max_hops}) exceeded"
            logger.warning(self._failure_reason)

        # Check if all paths are stuck (no candidates)
        if not self._active_paths:
            self._done = True
            self._failure_reason = "No valid candidates found"

    def get_result(self) -> TraversalResult:
        """Return the final traversal result."""
        if self._completed_path:
            return TraversalResult(
                success=True,
                path=self._completed_path,
                hops_taken=self._completed_path.length(),
                pages_explored=len(self._all_visited),
            )
        else:
            best = self._active_paths[0] if self._active_paths else None
            return TraversalResult(
                success=False,
                path=best,
                hops_taken=best.length() if best else 0,
                pages_explored=len(self._all_visited),
                failure_reason=self._failure_reason or "Unknown failure",
            )

    def get_hop_log(self) -> list[dict]:
        """Return the full log of all hops taken."""
        return self._hop_log

    def _log_hop(self, from_page: str, to_page: str, score: float,
                 reasoning: str, path_length: int):
        self._hop_log.append({
            "hop": path_length,
            "from": from_page,
            "to": to_page,
            "score": round(score, 4),
            "reasoning": reasoning,
        })
        logger.info(f"Hop {path_length}: {from_page} → {to_page} (score: {score:.3f})")
        if reasoning:
            logger.info(f"  Reasoning: {reasoning}")

    def stats(self) -> dict:
        return {
            "strategy": self.strategy,
            "active_paths": len(self._active_paths),
            "pages_explored": len(self._all_visited),
            "best_path_length": self._active_paths[0].length() if self._active_paths else 0,
            "done": self._done,
        }
