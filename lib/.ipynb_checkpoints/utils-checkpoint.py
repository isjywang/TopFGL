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
    if args.alg == "topfgl":
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
    elif args.dataset == "PubMed":
        num_features = 500
        num_labels = 3
    elif args.dataset == 'Computers':
        num_features = 767
        num_labels = 10       
    elif args.dataset == 'Photo':
        num_features = 745
        num_labels = 8   
    elif args.dataset == 'Ogbn':
        num_features = 128
        num_labels = 40
    elif args.dataset == 'CS':
        num_features = 6805
        num_labels = 15    
    elif args.dataset == 'Physics':
        num_features = 8415
        num_labels = 5            
    elif args.dataset == 'CoraFull':
        num_features = 8710
        num_labels = 70  
    elif args.dataset == 'pokec':
        num_features = 65
        num_labels = 3 ## there exists some unlabeled node (label -1)
    elif args.dataset == 'penn94':
        num_features = 4814
        num_labels = 3 ## there exists some unlabeled node (label -1)
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
    
    ## topfgl settings
    args.k = 3 + int(min(args.clients,20)/6) if args.dataset in ['Cora','CiteSeer','CoraFull'] else 8

    ## for large-scale graphs
    args.repair_fre = 5 if args.dataset in ['Ogbn','penn94'] else args.repair_fre
    args.k = 12 if args.dataset in ["pokec","penn94"] else args.k
