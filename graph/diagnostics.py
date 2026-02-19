"""Structural graph diagnostics for navigation dataset evaluation."""

from __future__ import annotations

import random
from typing import Dict, Tuple

import networkx as nx


def giant_component_ratio(G: nx.DiGraph) -> Tuple[float, int]:
    """Return (ratio, size) of the largest (undirected) connected component."""
    n = G.number_of_nodes()
    if n == 0:
        return 0.0, 0
    UG = G.to_undirected()
    components = list(nx.connected_components(UG))
    if not components:
        return 0.0, 0
    largest = max(components, key=len)
    size = len(largest)
    return size / n, size


def average_degree(G: nx.DiGraph) -> float:
    """Average total degree (in+out) across nodes."""
    degrees = [deg for _, deg in G.degree()]
    return (sum(degrees) / len(degrees)) if degrees else 0.0


def degree_distribution_stats(G: nx.DiGraph) -> Dict[str, float]:
    """Return min/max/mean degree summary stats."""
    degrees = [deg for _, deg in G.degree()]
    if not degrees:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(degrees)),
        "max": float(max(degrees)),
        "mean": float(sum(degrees) / len(degrees)),
    }


def clustering_coefficient(G: nx.DiGraph) -> float:
    """Average clustering coefficient on the undirected projection."""
    if G.number_of_nodes() == 0:
        return 0.0
    UG = G.to_undirected()
    return float(nx.average_clustering(UG))


def _giant_undirected_subgraph(G: nx.DiGraph) -> nx.Graph:
    UG = G.to_undirected()
    if UG.number_of_nodes() == 0:
        return UG
    largest_nodes = max(nx.connected_components(UG), key=len)
    return UG.subgraph(largest_nodes).copy()


def average_shortest_path(G: nx.DiGraph) -> float:
    """Average shortest path length on the (undirected) giant component."""
    subgraph = _giant_undirected_subgraph(G)
    if subgraph.number_of_nodes() < 2:
        return 0.0
    return float(nx.average_shortest_path_length(subgraph))


def graph_diameter(G: nx.DiGraph) -> int:
    """Diameter on the (undirected) giant component."""
    subgraph = _giant_undirected_subgraph(G)
    if subgraph.number_of_nodes() < 2:
        return 0
    return int(nx.diameter(subgraph))


def random_reachability(G: nx.DiGraph, trials: int = 100) -> float:
    """Fraction of random ordered pairs (a,b) with a path a -> b in the directed graph."""
    nodes = list(G.nodes())
    if len(nodes) < 2 or trials <= 0:
        return 0.0
    trials = min(trials, len(nodes) * (len(nodes) - 1))

    success = 0
    for _ in range(trials):
        a, b = random.sample(nodes, 2)
        if nx.has_path(G, a, b):
            success += 1
    return success / trials

