from preprocess import load_cora_graph
import dgl
import torch
import torch.nn as nn
from dgl.nn import GraphConv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, out_feats)
        
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g,h)
        return h
    
def dot_predictor(z, edges):
    u = edges[:,0] 
    v = edges[:,1]
    y_hat = (z[u] * z[v]).sum(dim=1)
    return y_hat

def train(model, g_train, node_feats, optimizer, predictor, epochs=100, neg_samples=1):

    for epoch in range(epochs):
        model.train()
        # Encoding edges
        z = model(g_train, node_feats)

        # Getting possitive edges
        u, v = g_train.edges()
        possitive_edges = torch.stack([u, v], dim=1)
        possitive_score = predictor(z, possitive_edges)
        possitive_label = torch.ones_like(possitive_score)

        # Negative sampling
        num_negative = u.shape[0] * neg_samples
        negative_u = torch.randint(0, g_train.num_nodes(), (num_negative,))
        negative_v = torch.randint(0, g_train.num_nodes(), (num_negative,))
        negative_edges = torch.stack([negative_u, negative_v], dim=1)
        negative_score = predictor(z, negative_edges)
        negative_label = torch.zeros_like(negative_score)

        
        scores = torch.cat([possitive_score, negative_score])
        labels = torch.cat([possitive_label, negative_label])

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f}")

def evaluate(model, predictor, g, node_feats, test_positive_edges, test_negative_edges):
    model.eval()

    with torch.no_grad():
        z = model(g, node_feats )
        positive_edges = torch.tensor(test_positive_edges)
        negative_edges = torch.tensor(test_negative_edges)

        positive_score = predictor(z, positive_edges)
        negative_score = predictor(z, negative_edges)

        scores = torch.cat([positive_score, negative_score]).cpu().numpy()
        labels = torch.cat([
            torch.ones(len(positive_score)), 
            torch.zeros(len(negative_score))]).cpu().numpy()


        auc = roc_auc_score(labels, scores)

    return auc

G_train, test_positive_edges, test_negative_edges, G, g_dgl = load_cora_graph()

g_train_dgl = dgl.from_networkx(G_train)
g_train_dgl = dgl.add_self_loop(g_train_dgl)

node_list = list(G_train.nodes())
g_train_dgl.ndata["feat"] = g_dgl.ndata["feat"][torch.tensor(node_list)]

# Setting up encoder (GCN)
in_feats = g_train_dgl.ndata["feat"].shape[1]
h_feats = 64
out_feats = 64
encoder = GCN(in_feats=in_feats, h_feats=h_feats, out_feats=out_feats)

# Setting up optimizer

optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

# Train model

train(
    model = encoder,
    g_train = g_train_dgl,
    node_feats = g_train_dgl.ndata["feat"],
    optimizer = optimizer,
    predictor = dot_predictor,
    epochs = 100,
    neg_samples = 1
)

mapping = {}
positive_mapped_edges = []
negative_mapped_edges = []

for new_index, old_node in enumerate(G_train.nodes()):
    mapping[old_node] = new_index

for u, v in test_positive_edges:

    if u in mapping and v in mapping:
        u_i = mapping[u]
        v_i = mapping[v]
        positive_mapped_edges.append((u_i, v_i))

for u, v in test_negative_edges:
    if u in mapping and v in mapping:
        u_i = mapping[u]
        v_i = mapping[v]
        negative_mapped_edges.append((u_i, v_i))

# Evaluation
auc = evaluate(
    model = encoder,
    predictor = dot_predictor,
    g = g_train_dgl,
    node_feats = g_train_dgl.ndata["feat"],
    test_positive_edges = positive_mapped_edges,
    test_negative_edges = negative_mapped_edges
)

print("Test AUC:", auc)
