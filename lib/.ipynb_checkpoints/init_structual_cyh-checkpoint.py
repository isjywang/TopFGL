import numpy as np
import os
import torch
from torch_geometric.utils import to_networkx, degree, to_dense_adj, to_scipy_sparse_matrix
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy import sparse as sp
import dgl
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time

def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)    
    torch.save(data, fpath)
def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)    
    return torch.load(fpath, map_location=torch.device('cpu'))
def get_data(client_id,clients,dataset):
    return [
        torch_load(
            "/home/1005wjy/datasets/", 
           f'{dataset}_{"disjoint"}/{clients}/partition_{client_id}.pt'
        )['client_data']
    ]


def compute_degree_similarity(d_a, d_b):
    return 1 / (1 + abs(d_a - d_b))

def compute_jaccard_similarity(neighbors_a, neighbors_b):

    # 计算交集和并集的大小
    intersection = len(neighbors_a.intersection(neighbors_b))
    union = len(neighbors_a.union(neighbors_b))

    j_sim = 0.0 if union == 0 else intersection / float(union)
    return j_sim
def matrix_power_worker(M1, power, result_queue):
    """计算矩阵幂并返回对角线元素"""
    M_temp = M1
    for _ in range(power - 1):  # 减1是因为第一次幂已经在M1中
        M_temp = M_temp.dot(M1)
    result_queue.put((power, M_temp.diagonal()))

# def get_random_emb(g):

def get_random_emb(g):
    # 转换为CSR格式
    # A1 = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes).tocsr()
    # print(11111)
    
    # # 直接计算 D^(-1)
    # D1 = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()
    # print(22222)
    
    # # 使用CSR格式的对角矩阵
    # Dinv1 = sp.diags(D1, format='csr')
    # print(33333)
    
    # # 使用高效的稀疏矩阵乘法
    # M1 = A1.dot(Dinv1)
    # print(44444)
    
    # # 确保使用CSR格式
    # M1 = M1.tocsr()
    # print(55555)
    
    # # 预分配存储空间
    # n_rw = 16
    # SE1 = [None] * n_rw
    # SE1[0] = torch.from_numpy(M1.diagonal()).float()
    # print(66666)
    
    # # 预计算对角线索引
    # diag_indices = np.diag_indices(M1.shape[0])
    
    # # 批量计算幂
    # M_power = M1
    # for i in range(1, n_rw):
    #     print(77777)
    #     M_power = M_power.dot(M1)
    #     SE1[i] = torch.from_numpy(M_power[diag_indices]).float()

    # # 使用预分配的列表直接stack
    # random_emb1 = torch.stack(SE1, dim=-1).cpu()
    # return random_emb1
    ##########################################################
    # #random walk embedding  [orginal]
    # A1 = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
    # print(11111)
    # D1 = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()
    # print(22222)
    # Dinv1=sp.diags(D1)
    # print(33333)
    # RW1=A1*Dinv1
    # print(44444)
    # M1=RW1
    # print(55555)
    # SE1=[torch.from_numpy(M1.diagonal()).float()]
    # print(66666)
    # M_power1=M1
    
    # n_rw=16
    # for _ in range(n_rw-1):
    #     print(77777)
    #     M_power1=M_power1*M1
    #     SE1.append(torch.from_numpy(M_power1.diagonal()).float())
    # random_emb1 = torch.stack(SE1,dim=-1).cpu()
#######################################################################################
    # random walk embedding  [fast version]
    device = torch.device("cuda:0")
    A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
    A_indices = torch.LongTensor([A.row, A.col])  # Indices on CUDA
    A_values = torch.FloatTensor(A.data)  # Values on CUDA
    # Create the sparse tensor on CUDA
    A_tensor = torch.sparse_coo_tensor(indices=A_indices, values=A_values, size=(g.num_nodes, g.num_nodes)).to(device) 
    # Convert the degree vector D to a PyTorch tensor and move to CUDA
    D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).to(device)

    RW = A_tensor * D.unsqueeze(0)
    #RW = torch.mm(A_tensor, Dinv)
    M = RW
    SE = [M.to_dense().diagonal().to(device)]
    M_power = M
    n_rw = 16
    for _ in range(n_rw - 1):
        M_power = M_power @ M  # Matrix multiplication on CUDA
        SE.append(M_power.to_dense().diagonal().to(device))  # Store diagonal on CUDA
    random_emb = torch.stack(SE,dim=-1).cpu()
    print("random walk emb over")
    return random_emb
    


def init_structure_encoding_node(g):

    # SE_rw
    A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
    D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

    Dinv=sp.diags(D)
    RW=A*Dinv
    M=RW

    SE=[torch.from_numpy(M.diagonal()).float()]
    M_power=M
    n_rw, n_dg = 16, 16
    for _ in range(n_rw-1):
        M_power=M_power*M
        SE.append(torch.from_numpy(M_power.diagonal()).float())
    SE_rw=torch.stack(SE,dim=-1)

    # PE_degree
    g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, n_dg)
    SE_dg = torch.zeros([g.num_nodes, n_dg])
    for i in range(len(g_dg)):
        SE_dg[i,int(g_dg[i]-1)] = 1

    g['stc_enc'] = torch.cat([SE_rw, SE_dg], dim=1)

    return g
    
def init_graphrepair(g):
    random_emb = get_random_emb(g)
#################################################################################################
    #################################################################################################
    # Degree  similarity embedding
    # Jaccard similarity embedding
    
    # 1. 获取所有节点的度
    g_dg = degree(g.edge_index[0], num_nodes=g.num_nodes).numpy()
    
    # 2. 获取每个节点的邻居集合（考虑有向边，取并集）
    neighbors = {}
    for i in range(g.num_nodes):
        outgoing_neighbors = set(g.edge_index[1][g.edge_index[0] == i].numpy())  # 节点 i 作为起始节点的邻居
        incoming_neighbors = set(g.edge_index[0][g.edge_index[1] == i].numpy())  # 节点 i 作为目标节点的邻居
        neighbors[i] = outgoing_neighbors.union(incoming_neighbors)
    # 存储所有节点与其邻居之间的度相似性，避免重复计算
    similarities_d, similarities_j = {}, {}

    # 3. 遍历每对节点，仅计算一次相似性（避免冗余计算）
    for node in range(g.num_nodes):
        node_degree = g_dg[node]
        node_neighbors = neighbors[node]
        # 只计算 node 与其邻居之间的相似性，避免重复计算
        for neighbor in node_neighbors:
            # 计算相似性，只计算一边（避免重复计算）
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
    print(11111111111111111111111111111111)
    # 4. 找到全局的最大度相似性 S_max 和最小 S_min
    all_similarities_d = np.concatenate([list(node_sim.values()) for node_sim in similarities_d.values()])
    all_similarities_j = np.concatenate([list(node_sim.values()) for node_sim in similarities_j.values()])
    
    S_max_d, S_min_d = all_similarities_d.max(), all_similarities_d.min()
    S_max_j, S_min_j = all_similarities_j.max(), all_similarities_j.min()
    
    # 归一化度相似性
    normalized_similarities_d, normalized_similarities_j = {}, {}

    for node in similarities_d:
        normalized_similarities_d[node] = {}
        for neighbor, sim in similarities_d[node].items():
            normalized_similarities_d[node][neighbor] = (sim - S_min_d) / (S_max_d - S_min_d)
    for node in similarities_j:
        normalized_similarities_j[node] = {}
        for neighbor, sim in similarities_j[node].items():
            normalized_similarities_j[node][neighbor] = (sim - S_min_j) / (S_max_j - S_min_j)    
    print(22222222222222222222222222222222222)
    # 5. 计算每个节点的平均度相似性
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
    print(3333333333333333333333333333333333333)
    # 结果：每个节点的平均归一化度相似性
    sorted_similarity_values_d = [average_normalized_similarity_d[node] for node in range(g.num_nodes)]
    sorted_similarity_values_j = [average_normalized_similarity_j[node] for node in range(g.num_nodes)]
    
    degree_emb  = torch.tensor(sorted_similarity_values_d, dtype=torch.float32).unsqueeze(1)
    jaccard_emb = torch.tensor(sorted_similarity_values_j, dtype=torch.float32).unsqueeze(1)

    print("shape of 3 embeddings:",random_emb.shape,degree_emb.shape,jaccard_emb.shape)
    g['stc_enc'] = torch.cat([random_emb, degree_emb, jaccard_emb], dim = 1)
    return g

# # prepare for fedstar
# for dataset in ["Cora","CiteSeer","PubMed","CS","Physics","Ogbn","CoraFull"]:
#     for clients in [10,20,30]:
#         print(dataset," now clients num is ",clients)    
#         for c in range(clients):
#             data_path = "/home/1005wjy/datasets/"
#             client_graph = get_data(c,clients,dataset)[0]
#             print("client ",c," get data over")
#             client_graph = init_structure_encoding_node(client_graph)
#             torch_save(data_path, f'{dataset}_disjoint/{clients}/initfedstar_{c}.pt', {
#                     'client_data': client_graph,
#                     'client_id': c
#                 })
#             print("client ",c," over")
        
# ## prepare for fedcap
# num_labels_dict = {"Cora":7,"CoraFull":70,"CiteSeer":6,"PubMed":3,"CS":15,"Physics":5,"Ogbn":40,"Reddit2":41}

for dataset in ["Products"]:
    for clients in [5]:
        print("now clients num is ",clients)    
        for c in range(clients):
            data_path = "/home/1005wjy/datasets/"
            client_graph = get_data(c,clients,dataset)[0]
            print("client ",c," get data over")
            client_graph = init_graphrepair(client_graph)
            torch_save(data_path, f'{dataset}_disjoint/{clients}/init_{c}.pt', {
                    'client_data': client_graph,
                    'client_id': c
                })
            print("client ",c," over")

import os
os._exit(0)
