import torch
import torch.nn as nn
import dgl
import random
import numpy as np
import networkx as nx
from karateclub import Node2Vec
from dgl.data import CoraGraphDataset
from sklearn.metrics import roc_auc_score
import numpy as np

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

G_simple = nx.Graph(G_und) 

components = list(nx.connected_components(G_simple))
giant = max(components, key=len)
print("Original nodes:", g.num_nodes(), "Original edges:", g.num_edges())
print("Number of components (undirected):", len(components))
print("Largest component size:", len(giant))

G = G_simple.subgraph(giant).copy()

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
    # Check if random edge choice is not a bridge and both nodes have degree > 1
    if not is_bridge((u,v), G_train) and G_train.degree(u) > 1 and G_train.degree(v) > 1:
        G_train.remove_edge(u, v)
        edges_removed.append((u,v))
    if len(edges_removed) == num_to_remove:
        break

assert nx.is_connected(G_train), "Training graph is disconnected"

# Test edges
test_positive_edges = edges_removed
nodes = list(G_train.nodes())

test_negative_edges = []

for _ in range(num_to_remove):
    u,v = random.sample(nodes, 2)

    if not G_train.has_edge(u,v):
        test_negative_edges.append((u,v))

all_test_edges = test_positive_edges + test_negative_edges

labels = [1]*len(test_positive_edges) + [0]*len(test_negative_edges)

common_neighboors_scores = []

for u,v in all_test_edges:
    common_neighboors = list(nx.common_neighbors(G_train, u,v))
    score = len(common_neighboors)
    common_neighboors_scores.append(score)

aa_index_scores = []

aa_index = list(nx.adamic_adar_index(G_train, all_test_edges))
for u,v,score in aa_index:
    aa_index_scores.append(score)

jaccard_scores = []

jaccard = list(nx.jaccard_coefficient(G_train, all_test_edges))
for u,v,score in jaccard:
    jaccard_scores.append(score)

common_neighboors_auc = roc_auc_score(labels, common_neighboors_scores)
aa_index_auc = roc_auc_score(labels, aa_index_scores)
jaccard_auc = roc_auc_score(labels, jaccard_scores)

print("Common Neighbors AUC:", common_neighboors_auc)
print("Adamic-Adar AUC:", aa_index_auc)
print("Jaccard AUC:", jaccard_auc)

# Shallow embeddings
model = Node2Vec(dimensions=128,walk_length=80,walk_number=10,p=1,q=1)

# Reindexing
mapping = {}

for new_index, old_node in enumerate(G.nodes()):
    mapping[old_node] = new_index

G_reindexed = nx.relabel_nodes(G, mapping)

model.fit(G_reindexed)
embeddings = model.get_embedding()
print(embeddings)

# Hadamard product for test edges
H_positive_products = []
H_negative_products = []

for u,v in  test_positive_edges:
    u_i = mapping[u]
    v_i = mapping[v]
    positive_edge_product = embeddings[u_i] * embeddings[v_i]
    H_positive_products.append(positive_edge_product)

for u,v in test_negative_edges:
    u_i = mapping[u]
    v_i = mapping[v]
    negative_edge_product = embeddings[u_i] * embeddings[v_i]
    H_negative_products.append(negative_edge_product)

# Test set
X_test = np.array(H_positive_products + H_negative_products, dtype=np.float32)
X_test = torch.tensor(X_test)
print("X_test: ",X_test)

Y_test = np.array([1]*len(H_positive_products) + [0]*len(H_negative_products), dtype=np.float32)
Y_test = torch.tensor(Y_test).unsqueeze(1)
print("Y_test : ",Y_test)

# Training edges
train_positive_edges = list(G_train.edges())
train_nodes = list(G_train.nodes())
train_negative_edges = []

while len(train_negative_edges) < len(train_positive_edges):
    u,v = random.sample(train_nodes, 2)
    if not G_train.has_edge(u,v): 
        train_negative_edges.append((u, v))

# Hadamard product for training edges
x_train = []
y_train = []

for u,v in train_positive_edges:
    u_i = mapping[u]
    v_i = mapping[v]
    x_train_product = embeddings[u_i] * embeddings[v_i]
    x_train.append(x_train_product)
    y_train.append(1)

for u,v in train_negative_edges:
    u_i = mapping[u]
    v_i = mapping[v]
    x_train_product = embeddings[u_i] * embeddings[v_i]
    x_train.append(x_train_product)
    y_train.append(0)

# Train set
X_train = np.array(x_train, dtype=np.float32)
X_train = torch.tensor(X_train)
print("X_train: ",X_train)

Y_train = np.array(y_train, dtype=np.float32)
Y_train = torch.tensor(Y_train).unsqueeze(1)
print("Y_train: ",Y_train)

mlp = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1))

loss = nn.BCEWithLogitsLoss()


