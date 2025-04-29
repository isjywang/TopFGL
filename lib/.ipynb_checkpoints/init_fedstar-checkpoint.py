import numpy as np
import os
import torch
from torch_geometric.utils import to_networkx, degree, to_dense_adj, to_scipy_sparse_matrix
import torch.nn.functional as F
from scipy import sparse as sp
import dgl
from queue import Queue
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
            "/home/1005wjy/datasets/", 
           f'{dataset}_{mode}/{clients}/partition_{client_id}.pt'
        )['client_data']
    ]


def matrix_power_worker(M1, power, result_queue):
    """计算矩阵幂并返回对角线元素"""
    M_temp = M1
    for _ in range(power - 1):  # 减1是因为第一次幂已经在M1中
        M_temp = M_temp.dot(M1)
    result_queue.put((power, M_temp.diagonal()))

# def get_random_emb(g):
def get_random_emb_node(g,c):
    n_rw=16
    device='cuda:1'
    # A1 = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
    # D1 = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()
    # Dinv1=sp.diags(D1)
    # RW1=A1*Dinv1
    # M1=RW1
    # SE1=[torch.from_numpy(M1.diagonal()).float()]
    # M_power1=M1
    # for _ in range(n_rw-1):
    #     M_power1=M_power1*M1
    #     SE1.append(torch.from_numpy(M_power1.diagonal()).float())
    # random_emb1 = torch.stack(SE1,dim=-1).cpu()
#######################################################################################################
    A1 = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
    D1 = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()
    Dinv1 = sp.diags(D1)
    RW1 = (A1 @ Dinv1).tocoo()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    indices = torch.tensor([RW1.row, RW1.col], dtype=torch.long, device=device)
    values = torch.tensor(RW1.data, dtype=torch.float32, device=device)
    RW_tensor = torch.sparse_coo_tensor(indices, values, (g.num_nodes, g.num_nodes)).coalesce()


    random_walk_embs = torch.zeros((g.num_nodes, n_rw), device=device)
    M_power = RW_tensor.clone()
    block_size = 10
    print("g.num_nodes:",g.num_nodes)
    
    for step in range(n_rw):
        print(step,c)
        for node_id in range(g.num_nodes):
            
            random_walk_embs[node_id, step] = M_power[node_id, node_id]
    
        M_power_new_indices = []
        M_power_new_values = []
    
        for start_row in range(0, g.num_nodes, block_size):
            end_row = min(start_row + block_size, g.num_nodes)
            row_indices = torch.arange(start_row, end_row, device=device)
            
            sub_M_power = M_power.index_select(0, row_indices)
            
            with torch.no_grad():  # 禁用梯度跟踪
                sub_result = torch.sparse.mm(sub_M_power, RW_tensor)
    
            coalesced = sub_result.coalesce()
            M_power_new_indices.append(coalesced.indices())
            M_power_new_values.append(coalesced.values())
    
            del sub_M_power, sub_result, coalesced
            torch.cuda.empty_cache()
    
        M_power_indices = torch.cat(M_power_new_indices, dim=1)
        M_power_values = torch.cat(M_power_new_values, dim=0)
        M_power = torch.sparse_coo_tensor(M_power_indices, M_power_values, (g.num_nodes, g.num_nodes)).coalesce()
    
        del M_power_new_indices, M_power_new_values, M_power_indices, M_power_values
        torch.cuda.empty_cache()
    
    random_walk_embs = random_walk_embs.cpu()
    # print("check",torch.equal(random_emb1,random_walk_embs),torch.allclose(random_emb1, random_walk_embs, atol=1e-03))
    # print(random_emb1)
    # print(random_walk_embs)
    return random_walk_embs
    # return random_emb1   
    
def get_random_emb(g):
    # # 转换为CSR格式
    # A1 = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes).tocsr()
    # # 直接计算 D^(-1)
    # D1 = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()
    # # 使用CSR格式的对角矩阵
    # Dinv1 = sp.diags(D1, format='csr')
    # # 使用高效的稀疏矩阵乘法
    # M1 = A1.dot(Dinv1)
    # # 确保使用CSR格式
    # M1 = M1.tocsr()
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
    #     M_power = M_power.dot(M1)
    #     SE1[i] = torch.from_numpy(M_power[diag_indices]).float().squeeze(0)

    # # # 使用预分配的列表直接stack
    # random_emb_new = torch.stack(SE1, dim=-1).cpu()
    # return random_emb_new
    ##########################################################
    # #random walk embedding  [orginal]
    # A1 = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)

    # D1 = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

    # Dinv1=sp.diags(D1)

    # RW1=A1*Dinv1

    # M1=RW1

    # SE1=[torch.from_numpy(M1.diagonal()).float()]

    # M_power1=M1
    
    # n_rw=16
    # for _ in range(n_rw-1):

    #     M_power1=M_power1*M1
    #     SE1.append(torch.from_numpy(M_power1.diagonal()).float())
    # random_emb1 = torch.stack(SE1,dim=-1).cpu()
    # return random_emb1
    # return random_emb_new
#######################################################################################
    # random walk embedding  [fast version]
    device = torch.device("cuda:4")
    A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
    A_indices = torch.LongTensor([A.row, A.col]).to(device)  # Indices on CUDA
    A_values = torch.FloatTensor(A.data).to(device)  # Values on CUDA
    # Create the sparse tensor on CUDA
    A_tensor = torch.sparse_coo_tensor(indices=A_indices, values=A_values, size=(g.num_nodes, g.num_nodes))
    # Convert the degree vector D to a PyTorch tensor and move to CUDA
    D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).to(device)

    Dinv = torch.diag(D)
    RW = A_tensor @ Dinv
    M = RW
    SE = [M.diagonal().to(device)]
    M_power = M
    n_rw = 16
    # n_rw = 4
    for _ in range(n_rw - 1):
        M_power = M_power @ M  # Matrix multiplication on CUDA
        SE.append(M_power.diagonal().to(device))  # Store diagonal on CUDA
    random_emb = torch.stack(SE,dim=-1).cpu()
    # print("check",torch.equal(random_emb1,random_emb),torch.allclose(random_emb1,random_emb, atol=1e-08))
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



mode = "disjoint"
for dataset in ['Computers','Photo','PubMed']:
    for clients in [3,5,8,13,15]:
        print(dataset," now clients num is ",clients)    
        for c in range(clients):
            data_path = "/home/1005wjy/datasets/"
            client_graph = get_data(c,clients,dataset,mode)[0]
            print("client ",c," get data over")
            client_graph = init_structure_encoding_node(client_graph)
            torch_save(data_path, f'{dataset}_{mode}/{clients}/initfedstar_{c}.pt', {
                    'client_data': client_graph,
                    'client_id': c
                })
            print("client ",c," over")
