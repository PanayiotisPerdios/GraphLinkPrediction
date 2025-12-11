import torch
import dgl
import random
import networkx as nx
from karateclub import Node2Vec
from dgl.data import CoraGraphDataset

def is_bridge(edge, G: nx.Graph):
    u,v = edge
    G.remove_edge(u,v)
    connected = nx.is_connected(G)
    G.add_edge(u, v)
    return not connected
    

dataset = CoraGraphDataset()
g = dataset[0]

nx_g = g.to_networkx()
G_und = nx_g.to_undirected()

components = list(nx.connected_components(G_und))
giant = max(components, key=len)
print("Original nodes:", g.num_nodes(), "Original edges:", g.num_edges())
print("Number of components (undirected):", len(components))
print("Largest component size:", len(giant))

G = G_und.subgraph(giant).copy()

print("Nodes in largest component (undirected):", G.number_of_nodes())
print("Edges in largest component (undirected):", G.number_of_edges())

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

#check if candidate edges are bridges
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
    # Check if random edge choice is not a bridge and both nodes have degree > 1
    if not is_bridge((u,v), G_train) and G_train.degree(u) > 1 and G_train.degree(v) > 1:
        G_train.remove_edge(u, v)
        edges_removed.append((u,v))
    if len(edges_removed) == num_to_remove:
        break


assert nx.is_connected(G_train), "Training graph is disconnected"

test_edges = edges_removed

