import numpy as np
import os
import torch
from torch_geometric.utils import to_networkx, degree
import torch.nn.functional as F
from scipy import sparse as sp
import dgl
import torch.optim as optim
from torch_geometric.nn import Node2Vec

def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)    
    torch.save(data, fpath)
    
def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)    
    return torch.load(fpath, map_location=torch.device('cpu'))
    
def get_data(client_id,clients,dataset,mode):
    return [
        torch_load(
            "../datasets/", 
           f'{dataset}_{mode}/{clients}/partition_{client_id}.pt'
        )['client_data']
    ]


def compute_degree_similarity(d_a, d_b):
    return 1 / (1 + abs(d_a - d_b))

def compute_jaccard_similarity(neighbors_a, neighbors_b):
    intersection = len(neighbors_a.intersection(neighbors_b))
    union = len(neighbors_a.union(neighbors_b))
    j_sim = 0.0 if union == 0 else intersection / float(union)
    return j_sim


def compute_node2vec_embeddings(g, embedding_dim=8, walk_length=6, context_size=6,
                                walks_per_node=6, p=1.0, q=1.0, num_negative_samples=6,
                                epochs=20, batch_size=128, lr=0.1, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Node2Vec(
        edge_index=g.edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        p=p,
        q=q,
        num_negative_samples=num_negative_samples,
        num_nodes=g.num_nodes, 
        sparse=True 
    ).to(device)

    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=4 if device=='cpu' else 0)

    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)

    model.eval()
    with torch.no_grad():
        RWE = model(torch.arange(g.num_nodes, device=device)).cpu()

    return RWE



def init_topfgl(g):

    ## Random walk embedding
    random_emb = compute_node2vec_embeddings(g)
    
    # Degree similarity embedding  and  Jaccard similarity embedding
    g_dg = degree(g.edge_index[0], num_nodes=g.num_nodes).numpy()
    
    neighbors = {}
    for i in range(g.num_nodes):
        outgoing_neighbors = set(g.edge_index[1][g.edge_index[0] == i].numpy())  
        incoming_neighbors = set(g.edge_index[0][g.edge_index[1] == i].numpy())  
        neighbors[i] = outgoing_neighbors.union(incoming_neighbors)

    similarities_d, similarities_j = {}, {}

    for node in range(g.num_nodes):
        node_degree = g_dg[node]
        node_neighbors = neighbors[node]

        for neighbor in node_neighbors:
            if node < neighbor:
                ## 1. compute degree similarity
                neighbor_degree = g_dg[neighbor]
                similarity = compute_degree_similarity(node_degree, neighbor_degree)
                if node not in similarities_d:
                    similarities_d[node] = {}
                similarities_d[node][neighbor] = similarity
                if neighbor not in similarities_d:
                    similarities_d[neighbor] = {}
                similarities_d[neighbor][node] = similarity

                ## 2. compute jaccard similarity
                similarity2 = compute_jaccard_similarity(neighbors[node], neighbors[neighbor])
                if node not in similarities_j:
                    similarities_j[node] = {}
                similarities_j[node][neighbor] = similarity2
                if neighbor not in similarities_j:
                    similarities_j[neighbor] = {}
                similarities_j[neighbor][node] = similarity2


    all_similarities_d = np.concatenate([list(node_sim.values()) for node_sim in similarities_d.values()])
    all_similarities_j = np.concatenate([list(node_sim.values()) for node_sim in similarities_j.values()])
    
    S_max_d, S_min_d = all_similarities_d.max(), all_similarities_d.min()
    S_max_j, S_min_j = all_similarities_j.max(), all_similarities_j.min()
    
    normalized_similarities_d, normalized_similarities_j = {}, {}

    for node in similarities_d:
        normalized_similarities_d[node] = {}
        for neighbor, sim in similarities_d[node].items():
            normalized_similarities_d[node][neighbor] = (sim - S_min_d) / (S_max_d - S_min_d)
    for node in similarities_j:
        normalized_similarities_j[node] = {}
        for neighbor, sim in similarities_j[node].items():
            normalized_similarities_j[node][neighbor] = (sim - S_min_j) / (S_max_j - S_min_j)    

    average_normalized_similarity_d, average_normalized_similarity_j = {}, {}
    for node in range(g.num_nodes):
        if node in normalized_similarities_d:
            average_normalized_similarity_d[node] = np.mean(list(normalized_similarities_d[node].values()))
        else:
            average_normalized_similarity_d[node] = 0
            
        if node in normalized_similarities_j:
            average_normalized_similarity_j[node] = np.mean(list(normalized_similarities_j[node].values()))
        else:
            average_normalized_similarity_j[node] = 0

    sorted_similarity_values_d = [average_normalized_similarity_d[node] for node in range(g.num_nodes)]
    sorted_similarity_values_j = [average_normalized_similarity_j[node] for node in range(g.num_nodes)]
    
    degree_emb  = torch.tensor(sorted_similarity_values_d, dtype=torch.float32).unsqueeze(1)
    jaccard_emb = torch.tensor(sorted_similarity_values_j, dtype=torch.float32).unsqueeze(1)


    g['stc_enc'] = torch.cat([random_emb, degree_emb, jaccard_emb], dim = 1)
    return g

        
# prepare for topfgl -- topological embedding init, run before the training.
# for mode in ['disjoint']:
#     for dataset in ['Cora']:
#         for clients in [10,20]:
#             print("now clients num is ",clients)    
#             for c in range(clients):
#                 data_path = "../datasets/"
#                 client_graph = get_data(c,clients,dataset,mode)[0]
#                 print("client ",c," get data over")
#                 client_graph = init_topfgl(client_graph)
#                 torch_save(data_path, f'{dataset}_{mode}/{clients}/init_{c}.pt', {
#                         'client_data': client_graph,
#                         'client_id': c
#                     })
#                 print("client ",c," over")

#         print("dataset:",dataset," mode:",mode," over")

# import os
# os._exit(0)
