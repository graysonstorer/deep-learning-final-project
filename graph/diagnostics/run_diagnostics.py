"""CLI entry point to build graph and print diagnostics."""

from __future__ import annotations


def main() -> None:
    from graph.build_page_graph import build_page_graph
    from graph.diagnostics import metrics as diag

    G, id_to_title = build_page_graph()

    print("\nGRAPH SUMMARY")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    ratio, giant_size = diag.giant_component_ratio(G)

    print("\nCONNECTIVITY")
    print("Giant Component Ratio:", ratio)
    print("Giant Component Size:", giant_size)

    print("\nDEGREE METRICS")
    print("Average Degree:", diag.average_degree(G))
    print("Degree Stats:", diag.degree_distribution_stats(G))

    print("\nCLUSTERING")
    print("Average Clustering Coefficient:", diag.clustering_coefficient(G))

    print("\nPATH METRICS")
    print("Average Shortest Path:", diag.average_shortest_path(G))
    print("Graph Diameter:", diag.graph_diameter(G))

    print("\nREACHABILITY")
    print("Random Reachability:", diag.random_reachability(G, trials=200))


if __name__ == "__main__":
    main()

