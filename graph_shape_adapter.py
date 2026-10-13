"""Correct CHGNet 0.3's empty adjacency schema without adding any edges."""


def normalize_empty_atom_graph(graph):
    edges = graph.atom_graph
    if edges.numel() == 0 and edges.ndim == 1:
        # An adjacency list has two columns even when there are no rows.
        # reshape preserves its empty contents, dtype, device and connectivity.
        graph.atom_graph = edges.reshape(0, 2)
        return True
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Invalid nonempty atom graph: {tuple(edges.shape)}")
    return False
