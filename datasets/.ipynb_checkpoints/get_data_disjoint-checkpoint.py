import torch
import random
import numpy as np
import os
import metispy as metis

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_networkx, is_undirected
from utils import get_data, split_train, torch_save, split_train_reddit

data_path = './' ## change to your data path
ratio_train = 0.2
seed = 2025
clients = [10, 20]
D = ['Cora', 'CiteSeer'] 

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)    
    return torch.load(fpath, map_location=torch.device('cpu'))
    
def generate_data(dataset, n_clients):
    ## get data
    data = split_train(get_data(dataset, data_path), dataset, data_path, ratio_train, 'disjoint', n_clients)
    ## split data
    split_subgraphs_fast(n_clients, data, dataset)
        

def split_subgraphs_fast(n_clients, data, dataset):
    print(data.edge_index.dtype)
    print(data.edge_index.min(),data.edge_index.max())
    print( (data.edge_index[0] == data.edge_index[1]).any())
    print(f"Is the graph undirected? {is_undirected(data.edge_index, num_nodes=data.num_nodes)}")
    print(len(data.y))
    G = to_networkx(data)
    print("begin metis")        
    n_cuts, membership = metis.part_graph(G, n_clients)

    assert len(list(set(membership))) == n_clients
    print(f'Graph partition done using Metis. Number of partitions: {len(list(set(membership)))}, Number of cuts: {n_cuts}')
        
    edge_index = data.edge_index

    for client_id in range(n_clients):
        client_indices = np.where(np.array(membership) == client_id)[0]
        client_indices = torch.tensor(client_indices)
        client_num_nodes = len(client_indices)

        mask_0 = torch.isin(edge_index[0], client_indices) 
        mask_1 = torch.isin(edge_index[1], client_indices)
        mask = mask_0 & mask_1
        client_edge_index = edge_index[:, mask]

        client_node_mapping = {int(old_idx): new_idx for new_idx, old_idx in enumerate(client_indices)}
        mapped_edge_index = torch.tensor([
            [client_node_mapping[old_idx.item()] for old_idx in client_edge_index[0]],
            [client_node_mapping[old_idx.item()] for old_idx in client_edge_index[1]]
        ], dtype=torch.long)
        
        client_edge_index = mapped_edge_index

        client_edge_index = torch.tensor(client_edge_index, dtype=torch.long)
        client_num_edges = client_edge_index.shape[1]

        client_x = data.x[client_indices]
        client_y = data.y[client_indices]
        client_train_mask = data.train_mask[client_indices]
        client_val_mask = data.val_mask[client_indices]
        client_test_mask = data.test_mask[client_indices]

        client_data = Data(
            x = client_x,
            y = client_y,
            edge_index = client_edge_index,
            train_mask = client_train_mask,
            val_mask = client_val_mask,
            test_mask = client_test_mask
        )
        
        assert torch.sum(client_train_mask).item() > 0

        torch_save(data_path, f'{dataset}_disjoint/{n_clients}/partition_{client_id}.pt', {
            'client_data': client_data,
            'client_id': client_id
        })
        print(f'Client ID: {client_id}, Number of training nodes: {client_num_nodes}, Number of training edges: {client_num_edges}')

for dataset in D:
    for n_clients in clients:
        generate_data(dataset=dataset, n_clients=n_clients)





