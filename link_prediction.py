import torch
import torch.nn as nn
import dgl
import random
import numpy as np
import networkx as nx
from karateclub import Node2Vec
from dgl.data import CoraGraphDataset
from sklearn.metrics import roc_auc_score
import torch.optim as optim
from sklearn.model_selection import train_test_split


class MLP(nn.Module):
    def __init__(self,embedding_dimension):
        super(MLP,self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(embedding_dimension, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,x):
        return self.linear_relu_stack(x)

def evaluate(model, X, Y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits)
        auc = roc_auc_score(Y.cpu().numpy(), probs.cpu().numpy())
    return auc

def train(model, X_train, Y_train, X_val, Y_val, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train)
        loss = criterion(logits, Y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_probs = torch.sigmoid(val_logits)
            val_auc = roc_auc_score(Y_val.cpu().numpy(), val_probs.cpu().numpy())

        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")

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

# Test edges
test_positive_edges = edges_removed
nodes = list(G_train.nodes())

test_negative_edges = []

for _ in range(num_to_remove):
    u,v = random.sample(nodes, 2)

    if not G_train.has_edge(u,v):
        test_negative_edges.append((u,v))

all_test_edges = test_positive_edges + test_negative_edges

test_labels_heu = [1]*len(test_positive_edges) + [0]*len(test_negative_edges)

common_neighbors_scores = []

for u,v in all_test_edges:
    common_neighbors = list(nx.common_neighbors(G_train, u,v))
    score = len(common_neighbors)
    common_neighbors_scores.append(score)

aa_index_scores = []

aa_index = list(nx.adamic_adar_index(G_train, all_test_edges))
for u,v,score in aa_index:
    aa_index_scores.append(score)

jaccard_scores = []

jaccard = list(nx.jaccard_coefficient(G_train, all_test_edges))
for u,v,score in jaccard:
    jaccard_scores.append(score)

common_neighboors_auc = roc_auc_score(test_labels_heu, common_neighbors_scores)
aa_index_auc = roc_auc_score(test_labels_heu, aa_index_scores)
jaccard_auc = roc_auc_score(test_labels_heu, jaccard_scores)

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
positive_edge_embeddings = []
negative_edge_embeddings = []

for u,v in  test_positive_edges:
    u_i = mapping[u]
    v_i = mapping[v]
    positive_edge_product = embeddings[u_i] * embeddings[v_i]
    positive_edge_embeddings.append(positive_edge_product)

for u,v in test_negative_edges:
    u_i = mapping[u]
    v_i = mapping[v]
    negative_edge_product = embeddings[u_i] * embeddings[v_i]
    negative_edge_embeddings.append(negative_edge_product)

# Test set
X_test = np.array(positive_edge_embeddings + negative_edge_embeddings, dtype=np.float32)
X_test = torch.tensor(X_test)
#print("X_test: ",X_test)

test_labels = [1]*len(positive_edge_embeddings) + [0]*len(negative_edge_embeddings)
Y_test = np.array(test_labels, dtype=np.float32)
Y_test = torch.tensor(Y_test).unsqueeze(1)
#print("Y_test : ",Y_test)

# Training edges
train_positive_edges = list(G_train.edges())
train_nodes = list(G_train.nodes())
train_negative_edges = []

while len(train_negative_edges) < len(train_positive_edges):
    u,v = random.sample(train_nodes, 2)
    if not G_train.has_edge(u,v): 
        train_negative_edges.append((u, v))

# Evaluation and Train split sets

train_possitive, val_possitive = train_test_split(train_positive_edges, 
                                        test_size = 0.2,
                                        random_state = 42,
                                        shuffle = True)

train_negative, val_negative = train_test_split(train_negative_edges, 
                                        test_size = 0.2,
                                        random_state = 42,
                                        shuffle = True)
X_train = []
Y_train = []

for u,v in train_possitive:
    u_i = mapping[u]
    v_i = mapping[v]
    edge_embedding = embeddings[u_i] * embeddings[v_i]
    X_train.append(edge_embedding)
    Y_train.append(1)

for u,v in train_negative:
    u_i = mapping[u]
    v_i = mapping[v]
    edge_embedding = embeddings[u_i] * embeddings[v_i]
    X_train.append(edge_embedding)
    Y_train.append(0)

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)

# Shuffle

perm = torch.randperm(X_train.size(0))

X_train = X_train[perm]
Y_train = Y_train[perm]

X_val = []
Y_val = []

for u, v in val_possitive:
    u_i = mapping[u]
    v_i = mapping[v]
    edge_embedding = embeddings[u_i] * embeddings[v_i]
    X_val.append(edge_embedding)
    Y_val.append(1)

for u, v in val_negative:
    u_i = mapping[u]
    v_i = mapping[v]
    edge_embedding = embeddings[u_i] * embeddings[v_i]
    X_val.append(edge_embedding)
    Y_val.append(0)

X_val = torch.tensor(np.array(X_val), dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1)

# Shuffle
perm = torch.randperm(X_val.size(0))

X_val = X_val[perm]
Y_val = Y_val[perm]

# Setup model
embedding_dimension = embeddings.shape[1]
model_mlp = MLP(embedding_dimension=embedding_dimension)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_mlp.parameters(), lr = 0.001)

# Train model
train(model = model_mlp, 
      X_train = X_train, 
      Y_train = Y_train, 
      X_val = X_val, 
      Y_val = Y_val,
      criterion = criterion,
      optimizer = optimizer,
      epochs = 5
)

# Final evaluation
val_auc = evaluate(model_mlp, X_val, Y_val)
test_auc = evaluate(model_mlp, X_test, Y_test)

print("Final Val AUC:", val_auc)
print("Final Test AUC:", test_auc)