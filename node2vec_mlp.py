from preprocess import load_cora_graph
from karateclub import Node2Vec
import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import networkx as nx
import torch.optim as optim

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

def train(model, X_train, Y_train, X_val, Y_val, criterion, optimizer, epochs=100):
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

G_train, test_positive_edges, test_negative_edges, G, g_dgl = load_cora_graph()

model = Node2Vec(dimensions=128,walk_length=80,walk_number=10,p=1,q=1)

# Reindexing
mapping = {}

for new_index, old_node in enumerate(G.nodes()):
    mapping[old_node] = new_index

G_reindexed = nx.relabel_nodes(G, mapping)

model.fit(G_reindexed)
embeddings = model.get_embedding()

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
      epochs = 100
)

# Final evaluation
val_auc = evaluate(model_mlp, X_val, Y_val)
test_auc = evaluate(model_mlp, X_test, Y_test)

print("Final Val AUC:", val_auc)
print("Final Test AUC:", test_auc)