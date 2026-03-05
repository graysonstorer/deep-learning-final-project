import pickle
from pathlib import Path

import networkx as nx


GRAPH_PATH = Path("data/processed/page_graph.gpickle")


def load_graph(path: Path = GRAPH_PATH) -> nx.DiGraph:
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found at {path}. Run build_page_graph.py first.")

    # Prefer NetworkX loader if available; fallback to pickle (compatible with our writer).
    read_gpickle = getattr(nx, "read_gpickle", None)
    if callable(read_gpickle):
        return read_gpickle(path)

    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, nx.DiGraph):
        raise TypeError(f"Unexpected object type in {path}: {type(obj)}")
    return obj


def main() -> None:
    G = load_graph()

    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    # Check for self-loops
    self_loops = list(nx.selfloop_edges(G))
    print(f"Number of self-loops: {len(self_loops)}")

    # Check edge weights are numeric and positive
    invalid_weights = []
    for u, v, d in G.edges(data=True):
        w = d.get("weight", None)
        if not isinstance(w, (int, float)) or float(w) <= 0.0:
            invalid_weights.append((u, v, d))

    if invalid_weights:
        print("Edges with invalid weights found:")
        for u, v, d in invalid_weights[:50]:
            print(f"{u} -> {v}: {d}")
        if len(invalid_weights) > 50:
            print(f"... and {len(invalid_weights) - 50} more")
    else:
        print("All edge weights are valid positive numbers.")

    # Basic connectivity (undirected giant component ratio)
    if G.number_of_nodes() > 0:
        UG = G.to_undirected()
        comps = list(nx.connected_components(UG))
        largest = max(comps, key=len) if comps else set()
        ratio = (len(largest) / G.number_of_nodes()) if G.number_of_nodes() else 0.0
        print(f"Giant component size: {len(largest)}")
        print(f"Giant component ratio: {ratio:.3f}")

    # Optional: print first 10 nodes and their out-degrees
    print("Sample nodes and out-degrees:")
    for node in list(G.nodes())[:10]:
        print(f"Node {node}: out-degree {G.out_degree(node)}")


if __name__ == "__main__":
    main()

