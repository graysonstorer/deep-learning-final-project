# graph/graph_rag.py
# GraphRAG: Builds a local knowledge graph as the bot explores.
# Nodes = Wikipedia pages (with embeddings).
# Edges = links between pages.
# Community detection groups related pages for smarter re-ranking.

import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

try:
    import community as community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    logger.warning("python-louvain not installed. Community detection disabled.")


@dataclass
class GraphNode:
    title: str
    embedding: Optional[np.ndarray] = None
    lede: str = ""
    categories: list[str] = field(default_factory=list)
    community_id: int = -1
    visit_count: int = 0


class GraphRAG:
    """
    Retrieval-Augmented Generation over a local Wikipedia subgraph.

    As the bot explores, it:
      1. Adds visited pages as nodes (with embeddings)
      2. Adds links between pages as edges
      3. Periodically re-runs community detection
      4. Uses community membership and graph proximity to re-rank candidates

    Key scoring signals:
      - Same community as target → high bonus
      - Short graph distance to target → bonus
      - High-degree hub pages (many connections) → small bonus
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: dict[str, GraphNode] = {}
        self.target_title: Optional[str] = None
        self.target_embedding: Optional[np.ndarray] = None
        self._communities_dirty = True
        self._hop_count = 0

    def set_target(self, title: str, embedding: np.ndarray):
        """Register the target page."""
        self.target_title = title
        self.target_embedding = embedding
        self.add_node(title, embedding=embedding)

    def add_node(
        self,
        title: str,
        embedding: Optional[np.ndarray] = None,
        lede: str = "",
        categories: list[str] = None,
    ):
        """Add or update a page node in the graph."""
        if title not in self.nodes:
            self.nodes[title] = GraphNode(title=title, embedding=embedding,
                                          lede=lede, categories=categories or [])
            self.graph.add_node(title)
        else:
            node = self.nodes[title]
            if embedding is not None:
                node.embedding = embedding
            if lede:
                node.lede = lede
            if categories:
                node.categories = categories
        self._communities_dirty = True

    def add_edge(self, from_title: str, to_title: str):
        """Add a directed edge (link) between two pages."""
        # Ensure both nodes exist
        if from_title not in self.nodes:
            self.add_node(from_title)
        if to_title not in self.nodes:
            self.add_node(to_title)
        self.graph.add_edge(from_title, to_title)
        self._communities_dirty = True

    def add_page_links(self, from_title: str, link_titles: list[str]):
        """Add all outbound links from a page to the graph."""
        for link in link_titles:
            self.add_edge(from_title, link)

    def record_visit(self, title: str):
        """Mark a page as visited."""
        if title in self.nodes:
            self.nodes[title].visit_count += 1
        self._hop_count += 1

        # Re-run community detection every 5 hops
        if self._hop_count % 5 == 0:
            self._detect_communities()

    def _detect_communities(self):
        """Run Louvain community detection on the undirected projection of the graph."""
        if not LOUVAIN_AVAILABLE:
            return
        if len(self.graph.nodes) < config_min_community():
            return

        try:
            undirected = self.graph.to_undirected()
            partition = community_louvain.best_partition(undirected)
            for title, comm_id in partition.items():
                if title in self.nodes:
                    self.nodes[title].community_id = comm_id
            self._communities_dirty = False
            logger.debug(f"Community detection: {len(set(partition.values()))} communities found")
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")

    def get_community(self, title: str) -> int:
        """Return the community ID for a page, or -1 if unknown."""
        node = self.nodes.get(title)
        return node.community_id if node else -1

    def target_community(self) -> int:
        """Return the community ID of the target page."""
        if self.target_title:
            return self.get_community(self.target_title)
        return -1

    def graph_distance(self, from_title: str, to_title: str) -> Optional[int]:
        """
        Shortest path length in the graph between two pages.
        Returns None if no path exists in the current local graph.
        """
        try:
            return nx.shortest_path_length(self.graph, from_title, to_title)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def score_candidate(self, candidate_title: str) -> float:
        """
        GraphRAG score for a candidate link.
        Returns a score in [0, 1] range representing:
          - Community match with target
          - Graph proximity to target
          - Hub page bonus
        """
        import config
        score = 0.0

        # 1. Community match bonus
        if LOUVAIN_AVAILABLE:
            cand_comm = self.get_community(candidate_title)
            tgt_comm = self.target_community()
            if cand_comm != -1 and tgt_comm != -1 and cand_comm == tgt_comm:
                score += config.GRAPHRAG_COMMUNITY_WEIGHT

        # 2. Graph proximity: is this candidate a direct or near neighbor of target?
        if self.target_title:
            dist = self.graph_distance(candidate_title, self.target_title)
            if dist == 1:
                score += config.GRAPHRAG_NEIGHBOR_WEIGHT
            elif dist == 2:
                score += config.GRAPHRAG_NEIGHBOR_WEIGHT * 0.5
            elif dist == 3:
                score += config.GRAPHRAG_NEIGHBOR_WEIGHT * 0.25

        # 3. Embedding similarity to target (if we have both embeddings)
        cand_node = self.nodes.get(candidate_title)
        if (cand_node is not None and
                cand_node.embedding is not None and
                self.target_embedding is not None):
            sim = float(np.dot(cand_node.embedding, self.target_embedding))
            score += sim * 0.1  # small bonus, main embedding score comes from ranker

        return min(score, 1.0)

    def get_stats(self) -> dict:
        """Return graph statistics for logging/UI."""
        return {
            "nodes": len(self.nodes),
            "edges": self.graph.number_of_edges(),
            "communities": len(set(n.community_id for n in self.nodes.values()
                                   if n.community_id != -1)),
            "target": self.target_title,
        }


def config_min_community():
    try:
        import config
        return config.GRAPHRAG_MIN_COMMUNITY_SIZE
    except Exception:
        return 3
