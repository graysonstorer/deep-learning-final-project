# core/wiki_fetcher.py
# Wikipedia API wrapper: fetch pages, extract links, get lede paragraphs

import time
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache

import requests
from bs4 import BeautifulSoup

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

logger = logging.getLogger(__name__)


@dataclass
class WikiPage:
    title: str
    url: str
    lede: str                           # First N sentences of the article
    links: list[str] = field(default_factory=list)   # All outbound Wikipedia link titles
    categories: list[str] = field(default_factory=list)
    raw_html: str = ""


class WikiFetcher:
    """
    Fetches Wikipedia pages via the MediaWiki API.
    Extracts:
      - Lede paragraph (first N sentences, for embedding)
      - All outbound internal links (candidates for next hop)
      - Page categories
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.WIKI_USER_AGENT})
        self._last_request = 0.0

    def _throttle(self):
        """Respect Wikipedia's rate limits."""
        elapsed = time.time() - self._last_request
        if elapsed < config.WIKI_REQUEST_DELAY:
            time.sleep(config.WIKI_REQUEST_DELAY - elapsed)
        self._last_request = time.time()

    @lru_cache(maxsize=512)
    def fetch_page(self, title: str) -> Optional[WikiPage]:
        """
        Fetch a Wikipedia page by title.
        Returns a WikiPage or None if not found.
        """
        self._throttle()
        logger.debug(f"Fetching: {title}")

        # Step 1: Get parsed HTML from API
        params = {
            "action": "parse",
            "page": title,
            "prop": "text|links|categories",
            "format": "json",
            "redirects": True,
        }
        try:
            resp = self.session.get(config.WIKI_API_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch '{title}': {e}")
            return None

        if "error" in data:
            logger.warning(f"Wiki API error for '{title}': {data['error'].get('info', '')}")
            return None

        parse = data["parse"]
        real_title = parse["title"]
        raw_html = parse["text"]["*"]

        # Step 2: Extract links (internal Wikipedia pages only)
        links = []
        for link in parse.get("links", []):
            # ns=0 = main article namespace (skip Talk:, User:, File:, etc.)
            if link.get("ns", -1) == 0 and "*" in link:
                links.append(link["*"])

        # Step 3: Extract categories
        categories = [
            c["*"].replace("Category:", "")
            for c in parse.get("categories", [])
            if not c.get("hidden")
        ]

        # Step 4: Parse lede from HTML
        lede = self._extract_lede(raw_html, real_title)

        url = f"https://en.wikipedia.org/wiki/{real_title.replace(' ', '_')}"
        return WikiPage(
            title=real_title,
            url=url,
            lede=lede,
            links=links,
            categories=categories,
            raw_html=raw_html,
        )

    def _extract_lede(self, html: str, title: str) -> str:
        """
        Extract the lede (introductory paragraph) from Wikipedia HTML.
        Returns the first N sentences as a plain text string.
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove unwanted elements
        for tag in soup.find_all(["table", "sup", "span.mw-editsection", ".infobox",
                                  ".navbox", ".toc", "style", "script"]):
            tag.decompose()

        # Find first substantive paragraph (skip short ones like disambiguation notices)
        paragraphs = soup.find_all("p")
        lede_text = ""
        for p in paragraphs:
            text = p.get_text(separator=" ", strip=True)
            # Skip empty, very short, or coordinate-only paragraphs
            if len(text) > 80 and not text.startswith("Coordinates"):
                lede_text = text
                break

        if not lede_text:
            lede_text = title  # fallback

        # Take first N sentences
        sentences = re.split(r"(?<=[.!?])\s+", lede_text)
        lede = " ".join(sentences[:config.WIKI_LEDE_SENTENCES])

        # Clean up citation brackets like [1], [2], etc.
        lede = re.sub(r"\[\d+\]", "", lede).strip()

        return lede

    def page_exists(self, title: str) -> bool:
        """Quick check if a Wikipedia page exists."""
        self._throttle()
        params = {
            "action": "query",
            "titles": title,
            "format": "json",
        }
        try:
            resp = self.session.get(config.WIKI_API_URL, params=params, timeout=5)
            data = resp.json()
            pages = data["query"]["pages"]
            return "-1" not in pages
        except Exception:
            return False

    def search(self, query: str, limit: int = 5) -> list[str]:
        """Search Wikipedia and return matching page titles."""
        self._throttle()
        params = {
            "action": "opensearch",
            "search": query,
            "limit": limit,
            "format": "json",
        }
        try:
            resp = self.session.get(config.WIKI_API_URL, params=params, timeout=5)
            data = resp.json()
            return data[1] if len(data) > 1 else []
        except Exception:
            return []
