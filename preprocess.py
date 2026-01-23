from dgl.data import CoraGraphDataset
import networkx as nx
import torch
import random

def load_cora_graph():
    dataset = CoraGraphDataset()
    g_dgl = dataset[0]

    # Convert to undirected NetworkX graph
    nx_g = g_dgl.to_networkx()
    G_und = nx_g.to_undirected()
    G_simple = nx.Graph(G_und)

    # Get largest connected component
    components = list(nx.connected_components(G_simple))
    giant = max(components, key=len)
    G = G_simple.subgraph(giant).copy()

    # Check for self-loops and multiple edges for candidate edges
    edges = set()

    for u,v in G.edges():
        if u != v:
            edges.add((min(u, v), max(u, v)))

    # Get G's bridges
    bridges = set()
    for u,v in nx.bridges(G):
        edge = (min(u, v), max(u, v))
        bridges.add(edge)

    # Check if candidate edges are bridges
    candidate_edges = []

    for e in edges:
        if e not in bridges:
            candidate_edges.append(e)

    candidate_num_edges = len(candidate_edges)
    percentage_to_remove = 0.10

    num_to_remove = int(candidate_num_edges * percentage_to_remove)

    edges_removed = []

    G_train = G.copy()

    candidate_edges_shuffled = candidate_edges.copy()
    random.shuffle(candidate_edges_shuffled)

    for u, v in candidate_edges_shuffled:
        if len(edges_removed) == num_to_remove:
            break

        # Check degree first
        if G_train.degree(u) <= 1 or G_train.degree(v) <= 1:
            continue

        # Check if edge is currently a bridge
        if (u, v) in nx.bridges(G_train) or (v, u) in nx.bridges(G_train):
            continue

        G_train.remove_edge(u, v)
        edges_removed.append((u, v))

    assert nx.is_connected(G_train), "Training graph is disconnected"

    # Test positive edges
    test_positive_edges = edges_removed

    # Test negative edges
    nodes = list(G_train.nodes())
    test_negative_edges = []

    for _ in range(num_to_remove):
        u,v = random.sample(nodes, 2)

        if not G.has_edge(u,v):
            test_negative_edges.append((u,v))

    return G_train, test_positive_edges, test_negative_edges, G, g_dgl

