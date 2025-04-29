import torch
import dgl
import numpy as np
import networkx as nx
import os
import random

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)    
    return torch.load(fpath, map_location=torch.device('cpu'))

def get_data(args, client_id):
    return [
        torch_load(
            "../datasets/",
            f'{args.dataset}_{args.mode}/{args.clients}/partition_{client_id}.pt'
        )['client_data']
    ]
    
def get_data_init(args, client_id):
    if args.alg == "fedtop":
        print(f'{args.dataset}_{args.mode}/{args.clients}/init_{client_id}.pt')
        return [
        torch_load(
            "../datasets/",
            f'{args.dataset}_{args.mode}/{args.clients}/init_{client_id}.pt'
        )['client_data']
    ]
    else:
        print(f'{args.dataset}_{args.mode}/{args.clients}/partition_{client_id}.pt')
        return [
            torch_load(
                "../datasets/", 
                f'{args.dataset}_{args.mode}/{args.clients}/partition_{client_id}.pt'
            )['client_data']
        ]        
    
    
def load_FGL_data(args):
    ## load dataset
    splitedData = {}
    if args.dataset == "Cora":
        num_features = 1433
        num_labels = 7
    elif args.dataset == "CiteSeer":
        num_features = 3703
        num_labels = 6
    else:
        print("Please input the information of new dataset !")
        
    for c in range(args.clients):
        client_graph = get_data_init(args, c)[0]
        train_size = len(client_graph.y[client_graph.train_mask])  
        splitedData[c] = (client_graph, num_features, num_labels, train_size)
    return splitedData



def set_config(args):
    ## experimental settings
    args.weight_decay = 1e-8
    args.local_epoch = 4 if args.dataset in ['CoraFull', 'Photo', 'Computers'] else 2
    
    ## fedtop settings
    args.repair_fre = 5 if args.dataset in ['Ogbn','penn94'] else 3
    args.k = 3 + int(min(args.clients,20)/6) if args.dataset in ['Cora','CiteSeer','CoraFull'] else 8

    ## for large-client-nums
    args.k = 12 if args.dataset in ["pokec","penn94"] else args.k
