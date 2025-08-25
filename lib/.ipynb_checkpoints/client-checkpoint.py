import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph 
from torch_geometric.utils import to_networkx  
import torch_geometric.data as Data 
from torch_geometric.utils import to_undirected 
import networkx as nx  
import time
import numpy as np
import random 
import torch.nn as nn
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

def copy(target, source, keys):
    for name in keys:
        target[name].data = source[name].data.clone()

def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()
        
class Client():
    def __init__(self, model, client_id, train_size, client_graph, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id

        self.train_size = train_size
        self.client_graph = client_graph
        self.optimizer = optimizer
        self.args = args
        self.dataLoader = DataLoader(dataset=[self.client_graph], batch_size=args.batch_size, shuffle=False)
        
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}

    def download_from_server(self, args, server):
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()        

    def local_train(self, local_epoch):
        train_FGL(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device, self.id)

    def evaluate(self):
        acc = eval_FGL(self.model, self.dataLoader, self.args.device)
        return acc


class Client_topfgl():
    def __init__(self, model, client_id, train_size, client_graph, optimizer, args, topology_learner, structual_optimizer, num_labels):
        self.model = model.to(args.device)
        self.topology_learner = topology_learner.to(args.device)
        
        self.id = client_id
        self.train_size = train_size
        self.client_graph = client_graph
        self.args = args
        self.k = self.args.k
        
        self.optimizer = optimizer
        self.structual_optimizer = structual_optimizer

        self.dataLoader = DataLoader(dataset=[self.client_graph], batch_size=args.batch_size, shuffle=False)
        
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.W2 = {key: value for key, value in self.topology_learner.named_parameters()}

        self.aug_begin, self.aug_end = torch.tensor([]), torch.tensor([])
        self.node_begin = self.client_graph.edge_index[0].clone()
        self.node_end = self.client_graph.edge_index[1].clone()
        self.gnn_time = 0 
        self.str_time = 0
        self.ccs = args.ccs
        self.knn_method = args.knn_method
        
    def connected_component_size(self):  
        return int(len(self.client_graph.y)/self.ccs) 
        
    def topology_augmentation(self):
        self.topology_learner.eval()
        self.model.eval()

        with torch.no_grad():
            for _, batch in enumerate(self.dataLoader):
                batch.to(self.args.device)
                pred_fea = self.model(batch)
                emb_fea = pred_fea.clone().detach()
                pred_fea = torch.argmax(pred_fea, axis=1)
                pred_str = self.topology_learner(batch, emb_fea)
                embeddings  = pred_str.clone().detach()
                pred_str = torch.argmax(pred_str, axis=1)

        dif_node = torch.nonzero(pred_str != pred_fea).squeeze().cpu().numpy()


        if self.knn_method == 'kdtree':
            embeddings_np = embeddings.detach().cpu().numpy()
            nn_finder = NearestNeighbors(n_neighbors=self.k + 1, algorithm='kd_tree', metric='euclidean', n_jobs=-1)
            nn_finder.fit(embeddings_np)
            indices_np = nn_finder.kneighbors(embeddings_np, return_distance=False)
    
            target_nodes_np = indices_np[:, 1:].flatten()
            source_nodes_np = np.arange(embeddings.shape[0]).repeat(self.k)
            source_nodes = torch.from_numpy(source_nodes_np).to(embeddings.device)
            target_nodes = torch.from_numpy(target_nodes_np).to(embeddings.device)
            KNN_edge_index = torch.stack([source_nodes, target_nodes], dim=0)
            KNN_graph = Data.Data(x=self.client_graph.x, y=self.client_graph.y, edge_index=KNN_edge_index)

        else:
            ## knn_graph use pyg version
            KNN_edge_index = knn_graph(embeddings, k=self.k, loop=False)
            KNN_graph = Data.Data(x=self.client_graph.x, y=self.client_graph.y, edge_index=KNN_edge_index)
  

        ## find largest connected components
        G = to_networkx(KNN_graph) 
        G2 = G.to_undirected()
        largest_cc = list(max(nx.connected_components(G2), key=lambda x: self.connected_component_size()) )

        edge1 = self.client_graph.edge_index.t()
        edge2 = edge1.flip(1)
        edge3 = torch.cat((edge1, edge2), dim=0)
        edge3_set = torch.unique(edge3, dim=0)
        new_edge = KNN_edge_index.t()
        new_edge_set = torch.unique(new_edge, dim=0).cpu()

        edge3_set = edge3_set.tolist()
        edge3_set = set(map(tuple, edge3_set)) 
        mask = torch.tensor([tuple(row.tolist()) not in edge3_set for row in new_edge_set])
        if len(mask)== 0 :
            print("aug. but no edges, quit")
            return
        new_edge_check = new_edge_set[mask]

        min_edges = torch.min(new_edge_check, dim=1)[0]
        max_edges = torch.max(new_edge_check, dim=1)[0]
        sorted_new_edge_check = torch.stack((min_edges, max_edges), dim=1)    
   
     
        largest_in = np.zeros(len(self.client_graph.y), dtype=bool)
        dif_check = np.zeros(len(self.client_graph.y), dtype=bool)
        largest_in[largest_cc] = True
        dif_check[dif_node] = True
        
        ## 1th, find the edges:    
        sedges = np.array(sorted_new_edge_check)
        mask_largest_in = largest_in[sedges[:, 0]] & largest_in[sedges[:, 1]]
        mask_dif_check = dif_check[sedges[:, 0]] | dif_check[sedges[:, 1]]
        y_array = np.array(self.client_graph.y)
        same_class_mask = y_array[sedges[:, 0]] == y_array[sedges[:, 1]]

        mask_condition = mask_largest_in & mask_dif_check & same_class_mask
        add_edges_begin = sedges[mask_condition, 0]
        add_edges_end = sedges[mask_condition, 1]
      

        # 2th, random delete edges:
        if len(self.aug_begin)/len(self.node_begin)>1/2:
            distances = torch.norm(emb_fea[self.aug_begin.to(torch.int64)] - emb_fea[self.aug_end.to(torch.int64)], dim=1).cpu().numpy()
            drop_edges_index = torch.tensor(np.random.choice(len(self.aug_begin), size = int(len(self.aug_begin)*0.3), p=distances/np.sum(distances), replace=False))
            mask = torch.ones(self.aug_begin.size(0), dtype=torch.bool)
            mask[drop_edges_index] = False
            self.aug_begin, self.aug_end = self.aug_begin[mask], self.aug_end[mask] 


        ## 3th, add new edges
        self.aug_begin = torch.cat((self.aug_begin, torch.tensor(add_edges_begin)))
        self.aug_end =   torch.cat((self.aug_end,   torch.tensor(add_edges_end)))
        origin_begin, origin_end = self.node_begin.clone(), self.node_end.clone()

        new_begin = torch.cat((origin_begin, self.aug_begin))
        new_end = torch.cat((origin_end, self.aug_end))
        self.client_graph.edge_index = torch.stack((new_begin, new_end)).to(torch.int64)
        self.dataLoader = DataLoader(dataset=[self.client_graph], batch_size=self.args.batch_size, shuffle=False)

        
    def download_from_server(self, args, server):
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()

        for q in server.W2:
            self.W2[q].data = server.W2[q].data.clone()
            
    def download_from_server_topfgl(self, args, server):
        
        for k in server.all_W[self.id]:
            self.W[k].data = server.all_W[self.id][k].data.clone()
            
        for k in server.all_W2[self.id]:
            self.W2[k].data = server.all_W2[self.id][k].data.clone()

            
    def local_train(self, local_epoch):
        train_topfgl(self.model, self.topology_learner, self.dataLoader, self.optimizer, self.structual_optimizer, local_epoch, self.args.device, self.id)
        
        
    def evaluate(self): 
        acc = eval_FGL(self.model, self.dataLoader, self.args.device)
        return acc




def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])


def train_FGL(model, dataloaders, optimizer, local_epoch, device, client_id, is_less=False):
    acc_test = []
    for epoch in range(local_epoch):
        model.train()
        for _, batch in enumerate(dataloaders):
            optimizer.zero_grad()
            batch.to(device)
            pred = model(batch)
            loss = model.loss(pred[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
        acc_t = eval_FGL(model, dataloaders, device)
        acc_test.append(acc_t)


def train_topfgl(model, top_learner, dataloaders, opt, opt_tl, local_epoch, device, client_id):
    acc_test = []
    for epoch in range(local_epoch):
        model.train()
        top_learner.train()
        for _, batch in enumerate(dataloaders):
            opt.zero_grad()
            batch.to(device)
            pred = model(batch)
            loss = model.loss(pred[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            opt.step()     
            emb_main = model(batch)
            opt_tl.zero_grad()
            emb_a = emb_main.clone().detach()
            pred2 = top_learner(batch, emb_a)
            loss2 = top_learner.loss(pred2[batch.train_mask], batch.y[batch.train_mask])
            loss2.backward()
            opt_tl.step()


def eval_FGL(model, test_loader, device):
    model.eval()
    acc_sum = 0.
    n = 0
    for batch in test_loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)[batch.test_mask]
            label = batch.y[batch.test_mask]
            loss = model.loss(pred, label)
        acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
        n += len(label)
    return acc_sum/n


