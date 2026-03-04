#!/usr/bin/env python3
# bot.py — WikiGame Bot: main entry point
#
# Usage:
#   python bot.py --start "Pizza" --target "Alan Turing"
#   python bot.py --start "Jazz" --target "World War II" --strategy beam --beam-width 3
#   python bot.py --start "Octopus" --target "Internet" --no-llm --verbose

import argparse
import logging
import sys
import os
import time
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

import config

# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(level: str, verbose: bool):
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_level = logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_FILE),
        ]
    )

logger = logging.getLogger("bot")


# ── Color output helpers ──────────────────────────────────────────────────────

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    def green(s): return f"{Fore.GREEN}{s}{Style.RESET_ALL}"
    def yellow(s): return f"{Fore.YELLOW}{s}{Style.RESET_ALL}"
    def red(s): return f"{Fore.RED}{s}{Style.RESET_ALL}"
    def cyan(s): return f"{Fore.CYAN}{s}{Style.RESET_ALL}"
    def bold(s): return f"{Style.BRIGHT}{s}{Style.RESET_ALL}"
except ImportError:
    def green(s): return s
    def yellow(s): return s
    def red(s): return s
    def cyan(s): return s
    def bold(s): return s


# ── Main bot class ────────────────────────────────────────────────────────────

class WikiGameBot:
    """
    Orchestrates the full Wikipedia game pipeline.
    """

    def __init__(
        self,
        strategy: str = "beam",
        beam_width: int = 3,
        max_hops: int = 30,
        use_llm: bool = True,
        verbose: bool = False,
    ):
        self.strategy = strategy
        self.beam_width = beam_width
        self.max_hops = max_hops
        self.use_llm = use_llm
        self.verbose = verbose

        print(bold("\n🚀 Initializing WikiGame Bot...\n"))

        # Initialize all components
        print("  Loading Wikipedia fetcher...")
        from core.wiki_fetcher import WikiFetcher
        self.fetcher = WikiFetcher()

        print("  Loading sentence embedder...")
        from core.embedder import Embedder
        self.embedder = Embedder()

        print("  Initializing GraphRAG...")
        from graph.graph_rag import GraphRAG
        self.graph_rag = GraphRAG()

        print("  Initializing WildGraph traversal engine...")
        from graph.wild_graph import WildGraph
        self.wild_graph = WildGraph(
            strategy=strategy,
            beam_width=beam_width,
            max_hops=max_hops,
        )

        print("  Loading GLiNER2 entity filter...")
        from search.gliner_filter import GlinerFilter
        self.gliner = GlinerFilter()

        # LLM setup
        self.llama = None
        if use_llm:
            print("  Connecting to Ollama (Llama)...")
            from llm.llama_agent import LlamaAgent
            self.llama = LlamaAgent()
            if self.llama.is_available():
                print(green(f"  ✓ Ollama connected ({config.OLLAMA_MODEL})"))
            else:
                print(yellow(f"  ⚠ Ollama not available — running in embedding-only mode"))
                self.llama = None

        # Scoring pipeline
        from search.ranker import Ranker
        self.ranker = Ranker(
            embedder=self.embedder,
            gliner_filter=self.gliner,
            graph_rag=self.graph_rag,
            llama_agent=self.llama,
        )

        print(green("\n  ✓ All components loaded\n"))

    def play(self, start: str, target: str) -> dict:
        """
        Play the Wikipedia game from start to target.
        Returns a result dict with path, hops, timing, etc.
        """
        print(bold(f"🎯 Target: {target}"))
        print(bold(f"📖 Start:  {start}\n"))

        game_start_time = time.time()

        # ── Fetch and validate start page ────────────────────────────────────
        print("Fetching start page...")
        start_page = self.fetcher.fetch_page(start)
        if not start_page:
            print(red(f"❌ Could not fetch start page: '{start}'"))
            # Try search
            results = self.fetcher.search(start, limit=3)
            if results:
                print(f"  Did you mean: {', '.join(results)}?")
            return {"success": False, "error": "Start page not found"}

        # ── Fetch and validate target page ───────────────────────────────────
        print("Fetching target page...")
        target_page = self.fetcher.fetch_page(target)
        if not target_page:
            print(red(f"❌ Could not fetch target page: '{target}'"))
            results = self.fetcher.search(target, limit=3)
            if results:
                print(f"  Did you mean: {', '.join(results)}?")
            return {"success": False, "error": "Target page not found"}

        # Use canonical titles from Wikipedia
        start_title = start_page.title
        target_title = target_page.title

        print(f"  Start  → {cyan(start_title)}")
        print(f"  Target → {cyan(target_title)}\n")

        # ── Check trivial case ────────────────────────────────────────────────
        if start_title.lower() == target_title.lower():
            print(green("✅ Start and target are the same page!"))
            return {"success": True, "path": [start_title], "hops": 0}

        # ── Set up ranker with target ─────────────────────────────────────────
        print("Setting up semantic target index...")
        self.ranker.set_target(target_title, target_page.lede)

        # Add target page to graph
        target_emb = self.embedder.embed(
            self.embedder.build_target_text(target_title, target_page.lede)
        )
        self.graph_rag.add_node(target_title, embedding=target_emb,
                                lede=target_page.lede,
                                categories=target_page.categories)
        self.graph_rag.add_page_links(target_title, target_page.links)

        # ── Initialize traversal ──────────────────────────────────────────────
        self.wild_graph.initialize(start_title, target_title)
        visited: set[str] = {start_title}

        # Add start page to graph
        start_emb = self.embedder.embed(
            self.embedder.build_target_text(start_title, start_page.lede)
        )
        self.graph_rag.add_node(start_title, embedding=start_emb,
                                lede=start_page.lede,
                                categories=start_page.categories)
        self.graph_rag.add_page_links(start_title, start_page.links)

        # Cache so we don't re-fetch
        page_cache = {start_title: start_page, target_title: target_page}

        print(f"Strategy: {bold(self.strategy)}, Beam width: {bold(str(self.beam_width))}\n")
        print("─" * 60)

        # ── Main traversal loop ───────────────────────────────────────────────
        hop_number = 0
        while not self.wild_graph.is_done():
            hop_number += 1
            frontier = self.wild_graph.get_frontier()

            if not frontier:
                print(red("❌ No frontier pages — stuck!"))
                break

            for current_title in frontier:
                # Fetch current page if not cached
                if current_title not in page_cache:
                    current_page = self.fetcher.fetch_page(current_title)
                    if not current_page:
                        logger.warning(f"Could not fetch '{current_title}', skipping")
                        continue
                    page_cache[current_title] = current_page
                else:
                    current_page = page_cache[current_title]

                # Add to graph
                cur_emb = self.embedder.embed(
                    self.embedder.build_target_text(current_title, current_page.lede)
                )
                self.graph_rag.add_node(current_title, embedding=cur_emb,
                                        lede=current_page.lede,
                                        categories=current_page.categories)
                self.graph_rag.add_page_links(current_title, current_page.links)
                self.graph_rag.record_visit(current_title)
                visited.add(current_title)

                links = current_page.links
                if not links:
                    logger.warning(f"'{current_title}' has no links!")
                    continue

                print(f"\nHop {hop_number} | Page: {cyan(current_title)} "
                      f"({len(links)} links)")

                # ── Rank candidates ───────────────────────────────────────────
                ranked = self.ranker.rank(
                    current_page_title=current_title,
                    current_page_lede=current_page.lede,
                    candidate_links=links,
                    visited_pages=visited,
                    verbose=self.verbose,
                )

                if not ranked:
                    print(yellow("  ⚠ No ranked candidates"))
                    continue

                # Show top candidates
                print(f"  Top candidates:")
                for i, (title, score, reasoning) in enumerate(ranked[:5], 1):
                    visited_tag = " (visited)" if title in visited else ""
                    print(f"    {i}. {title:45s} score={score:.3f}{visited_tag}")

                # Pick best candidate for this path
                best_title, best_score, reasoning = ranked[0]

                if reasoning:
                    print(f"  💭 Reasoning: {reasoning}")

                print(f"  → Choosing: {green(best_title)} (score: {best_score:.3f})")

                # Advance traversal
                self.wild_graph.advance(
                    from_page=current_title,
                    ranked_candidates=[(t, s) for t, s, _ in ranked],
                    reasoning=reasoning,
                )

                if self.wild_graph.is_done():
                    break

        # ── Results ───────────────────────────────────────────────────────────
        elapsed = time.time() - game_start_time
        result = self.wild_graph.get_result()

        print("\n" + "═" * 60)
        if result.success and result.path:
            path_str = " → ".join(result.path.pages)
            print(green(f"\n✅ SUCCESS! Reached '{target_title}' in {result.hops_taken} hops"))
            print(f"\nPath: {cyan(path_str)}")
            print(f"Pages explored: {result.pages_explored}")
            print(f"Time elapsed: {elapsed:.1f}s")
            print(f"Average hop score: {result.path.average_score():.3f}")

            # Optional: ask Llama to explain the path
            if self.llama and result.path:
                print("\n💭 Path explanation:")
                explanation = self.llama.summarize_path(result.path.pages, target_title)
                if explanation:
                    print(f"  {explanation}")
        else:
            best_path = result.path
            if best_path:
                path_str = " → ".join(best_path.pages[-5:])  # show last 5 pages
                print(red(f"\n❌ FAILED: {result.failure_reason}"))
                print(f"Best path (last 5 hops): ...{cyan(path_str)}")
            else:
                print(red(f"\n❌ FAILED: {result.failure_reason}"))
            print(f"Hops taken: {result.hops_taken}")
            print(f"Pages explored: {result.pages_explored}")

        return {
            "success": result.success,
            "path": result.path.pages if result.path else [],
            "hops": result.hops_taken,
            "pages_explored": result.pages_explored,
            "elapsed_seconds": elapsed,
            "hop_log": self.wild_graph.get_hop_log(),
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="WikiGame Bot — navigate Wikipedia with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bot.py --start "Pizza" --target "Alan Turing"
  python bot.py --start "Jazz" --target "World War II" --strategy beam
  python bot.py --start "Octopus" --target "Internet" --no-llm --verbose
  python bot.py --start "Mount Everest" --target "Calculus" --beam-width 5
        """
    )
    parser.add_argument("--start", required=True, help="Starting Wikipedia page title")
    parser.add_argument("--target", required=True, help="Target Wikipedia page title")
    parser.add_argument("--strategy", choices=["greedy", "beam"], default="beam",
                        help="Traversal strategy (default: beam)")
    parser.add_argument("--beam-width", type=int, default=config.BEAM_WIDTH,
                        help=f"Beam width for beam search (default: {config.BEAM_WIDTH})")
    parser.add_argument("--max-hops", type=int, default=config.MAX_HOPS,
                        help=f"Max hops before giving up (default: {config.MAX_HOPS})")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable Llama reranking (faster, pure embedding mode)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-candidate scores at each hop")
    parser.add_argument("--log-level", default=config.LOG_LEVEL,
                        choices=["DEBUG", "INFO", "WARNING"],
                        help="Log verbosity")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level, args.verbose)

    bot = WikiGameBot(
        strategy=args.strategy,
        beam_width=args.beam_width,
        max_hops=args.max_hops,
        use_llm=not args.no_llm,
        verbose=args.verbose,
    )

    result = bot.play(start=args.start, target=args.target)
    sys.exit(0 if result.get("success") else 1)
