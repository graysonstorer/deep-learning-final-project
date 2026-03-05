# llm/llama_agent.py
# Llama via Ollama: chain-of-thought reasoning for hop decisions.
#
# The LLM's job is NOT to pick from scratch — the embedding pipeline already
# narrows it to top-K good candidates. Llama's job is to apply higher-level
# reasoning: "which of these K options is strategically closest to the target?"

import logging
import re
import json
from typing import Optional

import requests

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert at the Wikipedia game — navigating between Wikipedia pages by clicking links.
Your job is to choose the BEST single link to click to get closer to the target page.
Think step by step about the semantic, historical, and conceptual connections between topics.
You must respond with ONLY a JSON object in exactly this format:
{"choice": <number>, "reasoning": "<one sentence explanation>"}
Do not include any other text."""


HOP_PROMPT_TEMPLATE = """You are navigating Wikipedia to reach: "{target}"
You are currently on: "{current}"

Here are your {n} best link candidates (pre-ranked by semantic similarity):
{candidates}

Which link gets you closest to "{target}"? 
Think about conceptual, historical, geographic, or field-of-study connections.
Respond with ONLY: {{"choice": <number 1-{n}>, "reasoning": "<one sentence>"}}"""


class LlamaAgent:
    """
    Wraps Ollama's local Llama API for hop decision-making.

    The agent receives the top-K pre-ranked candidates and picks the best one,
    with a one-sentence chain-of-thought explanation.
    """

    def __init__(self):
        import config
        self.host = config.OLLAMA_HOST
        self.model = config.OLLAMA_MODEL
        self.timeout = config.OLLAMA_TIMEOUT
        self.temperature = config.LLM_TEMPERATURE
        self._available = None  # cached availability check

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is loaded."""
        if self._available is not None:
            return self._available
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            data = resp.json()
            models = [m["name"].split(":")[0] for m in data.get("models", [])]
            self._available = self.model in models or any(
                self.model in m for m in models
            )
            if not self._available:
                logger.warning(
                    f"Ollama is running but model '{self.model}' not found. "
                    f"Available: {models}. Run: ollama pull {self.model}"
                )
            return self._available
        except requests.RequestException:
            logger.warning(
                f"Ollama not reachable at {self.host}. "
                f"Start it with: ollama serve"
            )
            self._available = False
            return False

    def _call(self, prompt: str) -> str:
        """Make a raw call to Ollama's generate API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 150,    # short response — we only need a number + sentence
            }
        }
        resp = requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def pick_best_hop(
        self,
        current_page: str,
        target_page: str,
        candidates: list[tuple[str, float]],  # (title, score) pre-ranked
    ) -> tuple[str, str]:
        """
        Ask Llama to pick the best candidate from the pre-ranked list.

        Returns (chosen_title, reasoning_string).
        Falls back to the top embedding candidate if LLM fails.
        """
        if not candidates:
            return "", "No candidates"

        # Format candidate list for the prompt
        candidate_lines = []
        for i, (title, score) in enumerate(candidates, 1):
            candidate_lines.append(f"  {i}. {title} (relevance: {score:.2f})")
        candidates_text = "\n".join(candidate_lines)

        prompt = HOP_PROMPT_TEMPLATE.format(
            target=target_page,
            current=current_page,
            n=len(candidates),
            candidates=candidates_text,
        )

        try:
            raw_response = self._call(prompt)
            choice_idx, reasoning = self._parse_response(raw_response, len(candidates))
            chosen_title = candidates[choice_idx - 1][0]
            return chosen_title, reasoning

        except Exception as e:
            logger.warning(f"LLM call failed: {e}. Defaulting to top embedding candidate.")
            return candidates[0][0], "LLM unavailable, used embedding ranking"

    def _parse_response(self, response: str, max_choice: int) -> tuple[int, str]:
        """
        Parse the LLM's JSON response.
        Returns (choice_index_1based, reasoning_string).
        """
        # Try JSON parsing first
        try:
            # Extract JSON even if there's surrounding text
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                choice = int(data.get("choice", 1))
                reasoning = str(data.get("reasoning", ""))
                choice = max(1, min(choice, max_choice))
                return choice, reasoning
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Fallback: extract any number from the response
        numbers = re.findall(r'\b([1-9]\d?)\b', response)
        for num_str in numbers:
            num = int(num_str)
            if 1 <= num <= max_choice:
                # Try to extract reasoning from the rest
                reasoning = re.sub(r'\{.*?\}', '', response).strip()
                reasoning = reasoning[:150] if reasoning else "LLM response parsed"
                return num, reasoning

        logger.warning(f"Could not parse LLM response: {response[:100]}")
        return 1, "Default: used top candidate"

    def summarize_path(self, path: list[str], target: str) -> str:
        """
        Optional: ask Llama to explain why the path makes sense.
        Good for debugging or showing users interesting explanations.
        """
        if not self.is_available():
            return ""

        path_str = " → ".join(path)
        prompt = (
            f"Explain in 2-3 sentences why this Wikipedia path makes logical sense:\n"
            f"{path_str}\n"
            f"(The goal was to reach '{target}')"
        )
        try:
            return self._call(prompt)
        except Exception:
            return ""
