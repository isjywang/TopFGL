import os
import torch
import numpy as np
import torch_geometric.datasets as datasets
import torch_geometric.transforms as T
import dgl

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected
from torch_geometric.utils import degree
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from scipy.io import loadmat
from torch_geometric.utils import one_hot

def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)    
    torch.save(data, fpath)

def get_data(dataset, data_path):
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        data = datasets.Planetoid(data_path, dataset, transform=T.NormalizeFeatures())[0]  
    elif dataset == 'CoraFull':
        data = datasets.CoraFull(data_path, transform=T.NormalizeFeatures())[0] 
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
    elif dataset in ['Computers', 'Photo']:
        data = datasets.Amazon(data_path, dataset, transform=T.NormalizeFeatures() )[0]
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
    elif dataset in ['Ogbn']:
        data = PygNodePropPredDataset('ogbn-arxiv', root=data_path, transform=T.ToUndirected())[0]
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
        data.y = data.y.view(-1) 
    elif dataset in ['CS','Physics']:
        data = datasets.Coauthor(data_path, dataset, transform=T.NormalizeFeatures())[0]
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
    elif dataset in ['penn94']:
        ## for the download, see https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/dataset.py#L233
        mat = loadmat('./Penn94.mat')
        A = mat['A'].tocsr().tocoo()
        row = torch.from_numpy(A.row).to(torch.long)
        col = torch.from_numpy(A.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
    
        metadata = torch.from_numpy(mat['local_info'].astype('int64'))
    
        xs = []
        y = metadata[:, 1] - 1  # gender label, -1 means unlabeled
        x = torch.cat([metadata[:, :1], metadata[:, 2:]], dim=-1)
        for i in range(x.size(1)):
            _, out = x[:, i].unique(return_inverse=True)
            xs.append(one_hot(out))
        x = torch.cat(xs, dim=-1)
    
        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
   
    elif dataset in ['pokec']:
        ## for the download, see https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/dataset.py#L233
        fulldata = loadmat('./pokec.mat')
        edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
        node_feat = torch.tensor(fulldata['node_feat']).float()
        label = fulldata['label'].flatten()
        print("label:",label)
        for i in range(len(label)):
            if label[i] == -1:
                label[i] = 2
        label = torch.tensor(label, dtype=torch.long)  # gender label, -1 means unlabeled
        edge_index = to_undirected(edge_index, num_nodes=len(label))
        
        data = Data(x=node_feat, edge_index=edge_index, y=label)
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
           
    else:
        print("please input correct dataset name!")
    return data


def split_train(data, dataset, data_path, ratio_train, mode, n_clients):
    n_data = data.num_nodes
    ratio_test = (1-ratio_train)/2
    n_train = round(n_data * ratio_train)
    n_test = round(n_data * ratio_test)
    
    data.train_mask.fill_(False)
    data.test_mask.fill_(False)
    data.val_mask.fill_(False)    
    
    permuted_indices = torch.randperm(n_data)        
    train_indices = permuted_indices[:n_train]
    test_indices = permuted_indices[n_train:n_train+n_test]
    val_indices = permuted_indices[n_train+n_test:]

    data.train_mask[train_indices] = True
    data.test_mask[test_indices] = True
    data.val_mask[val_indices] = True

    torch_save(data_path, f'{dataset}_{mode}/{n_clients}/train.pt', {'data': data})
    torch_save(data_path, f'{dataset}_{mode}/{n_clients}/test.pt', {'data': data})
    torch_save(data_path, f'{dataset}_{mode}/{n_clients}/val.pt', {'data': data})
    print(f'splition done, n_train: {n_train}, n_test: {n_test}, n_val: {len(val_indices)}')
    return data

