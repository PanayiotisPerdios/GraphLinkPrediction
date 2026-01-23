from preprocess import load_cora_graph
import networkx as nx
from sklearn.metrics import roc_auc_score

G_train, test_positive_edges, test_negative_edges, G, g_dgl = load_cora_graph()

all_test_edges = test_positive_edges + test_negative_edges

labels = [1]*len(test_positive_edges) + [0]*len(test_negative_edges)

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

common_neighbors_auc = roc_auc_score(labels, common_neighbors_scores)
aa_index_auc = roc_auc_score(labels, aa_index_scores)
jaccard_auc = roc_auc_score(labels, jaccard_scores)

print("Common Neighbors AUC:", common_neighbors_auc)
print("Adamic-Adar AUC:", aa_index_auc)
print("Jaccard AUC:", jaccard_auc)


