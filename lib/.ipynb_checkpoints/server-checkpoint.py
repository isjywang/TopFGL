import torch
import numpy as np
import random
import networkx as nx
from dtaidistance import dtw
from collections import defaultdict, OrderedDict
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from init_structural import init_topfgl
    
def from_networkx(G, group_node_attrs=None, group_edge_attrs=None):
    G = G.to_directed() if not nx.is_directed(G) else G
    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data = defaultdict(list)
    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}
    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in G.graph.items():
        if key == 'node_default' or key == 'edge_default':
            continue
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], torch.Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError, RuntimeError):
                pass

    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data



class Server():
    def __init__(self, model, device):
        self.model = model.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients):
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()




class Server_topfgl():
    def __init__(self, model, topology_learner, device, num_features, num_labels, args, all_model, all_learner):
        self.model = model.to(device)
        self.topology_learner = topology_learner.to(device)
        self.args = args
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.W2 = {key: value for key, value in self.topology_learner.named_parameters()}
        
        self.all_model, self.all_learner = all_model, all_learner ##personalized aggregated model
        self.all_W, self.all_W2 = [], []
        for i in range(self.args.clients):
            W = {key: value for key, value in self.all_model[i].named_parameters()}
            W2 = {key: value for key, value in self.all_learner[i].named_parameters()}
            self.all_W.append(W)
            self.all_W2.append(W2)
            
        self.num_features = num_features
        self.num_labels = num_labels
        self.model_cache = []
        self.device = device
        self.random_graph = self.build_random_graph()
        self.random_graph = init_topfgl(self.random_graph)

        
    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients):
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()
        for q in self.W2.keys():
            self.W2[q].data = torch.div(torch.sum(torch.stack([torch.mul(client.W2[q].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()
            
    def build_random_graph(self):
        num_nodes, num_graphs = 100, 1
        data = from_networkx(nx.random_partition_graph([num_nodes] * num_graphs, p_in=0.06, p_out=0, seed=666))
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, self.num_features))
        return data

    def get_embedding(self, selected_clients, data):
        all_emb = []
        data = data.to(self.device)
        for client in selected_clients:
            client.topology_learner.eval()
            client.model.eval()
            with torch.no_grad():
                emb_main = client.model(data)
                emb_main = emb_main.clone().detach()
                emb  = client.topology_learner(data, emb_main)
                emb      = emb.clone().detach()
                all_emb.append(emb)
        return all_emb

    def center(self, x):
        return x - x.mean(dim=0, keepdim=True) 

    def cka_similarity(self, x_a, x_b, epsilon=1e-9):

        n = x_a.size(0)
        if n != x_b.size(0):
             raise ValueError("Inputs x_a and x_b must have the same number of samples (dim 0)")

        x_a_centered = self.center(x_a)
        x_b_centered = self.center(x_b)
        c = x_a_centered.t() @ x_b_centered
        trace_cka = torch.sum(c * c)

        d_a = x_a_centered.t() @ x_a_centered 
        trace_kka_a = torch.sum(d_a * d_a) 
        d_b = x_b_centered.t() @ x_b_centered
        trace_kka_b = torch.sum(d_b * d_b)
        trace_kka = torch.sqrt(trace_kka_a * trace_kka_b + epsilon)

        cka = trace_cka / (trace_kka + epsilon)
        return cka

        
    def topfgl_aggregate(self, selected_clients):
        embedding = self.get_embedding(selected_clients, self.random_graph)
        num_clients = len(selected_clients)
        cka = torch.ones((num_clients, num_clients), device=self.device)
        
        for i in range(num_clients):
            for j in range(i+1,num_clients):
                this_cka = float(self.cka_similarity(embedding[i], embedding[j]))
                cka[i][j], cka[j][i] = this_cka, this_cka

        # selecting all clients
        for i in range(self.args.clients):
            for k in self.all_W[i].keys():
                self.all_W[i][k].data = torch.div( torch.sum(torch.stack([torch.mul(selected_clients[client].W[k].data, cka[i][client]) for client in range(len(selected_clients))]), dim=0)  , sum(cka[i]) ).clone()
                
            for k in self.all_W2[i].keys():
                self.all_W2[i][k].data = torch.div( torch.sum(torch.stack([torch.mul(selected_clients[client].W2[k].data, cka[i][client]) for client in range(len(selected_clients))]), dim=0)  , sum(cka[i]) ).clone()


