'''
Descripttion: 
version: 
Author: 成凯阳
Date: 2022-04-21 03:03:08
LastEditors: 成凯阳
LastEditTime: 2022-06-24 01:36:57
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
def create_var(tensor,requires_grad=None):
    if requires_grad is None:
        return Variable(tensor)
    else:
        return Variable(tensor, requires_grad=requires_grad)

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)
def unique_tensor(tensor):
    tensor = (tensor.data).cpu().numpy()
    unique_tensor = []
    visited = [-1 for _ in range(tensor.shape[0])]
    for i in range(tensor.shape[0] - 1):
        if visited[i] != -1: continue
        for j in range(i+1, tensor.shape[0]):
            if visited[j] != -1: continue
            boolean = np.allclose(tensor[i,:], tensor[j,:], atol=2e-07)
            if boolean:
                if visited[i] == -1:
                    unique_tensor.append(tensor[i,:])
                    visited[i] = len(unique_tensor) - 1
                
                visited[j] = len(unique_tensor) - 1
    
    for i in range(tensor.shape[0]):
        if visited[i] != -1: continue
        unique_tensor.append(tensor[i,:])
        visited[i] = len(unique_tensor) - 1

    unique_tensor = torch.tensor(np.stack(unique_tensor, axis=0)).cuda()
    return unique_tensor, visited
def avg_pool(all_vecs, scope, dim):
    size = create_var(torch.Tensor([le for _,le in scope]))
    return all_vecs.sum(dim=dim) / size.unsqueeze(-1)

def stack_pad_tensor(tensor_list):
    max_len = max([t.size(0) for t in tensor_list])
    for i,tensor in enumerate(tensor_list):
        pad_len = max_len - tensor.size(0)
        tensor_list[i] = F.pad( tensor, (0,0,0,pad_len) )
    return torch.stack(tensor_list, dim=0)
def create_pad_tensor(alist, extra_len=0):
    max_len = max([len(a) for a in alist]) + 1 + extra_len
    for a in alist:
        pad_len = max_len - len(a)
        a.extend([0] * pad_len)
    return torch.IntTensor(alist)
#3D padded tensor to 2D matrix, with padded zeros removed
def flatten_tensor(tensor, scope):
    assert tensor.size(0) == len(scope)
    tlist = []
    for i,tup in enumerate(scope):
        le = tup[1]
        tlist.append( tensor[i, 0:le] )
    return torch.cat(tlist, dim=0)

#2D matrix to 3D padded tensor
def inflate_tensor(tensor, scope): 
    max_len = max([le for _,le in scope])
    batch_vecs = []
    for st,le in scope:
        cur_vecs = tensor[st : st + le]
        cur_vecs = F.pad( cur_vecs, (0,0,0,max_len-le) )
        batch_vecs.append( cur_vecs )

    return torch.stack(batch_vecs, dim=0)

def GRU(x, h_nei, W_z, W_r, U_r, W_h):
    hidden_size = x.size()[-1]
    sum_h = h_nei.sum(dim=1)
    z_input = torch.cat([x,sum_h], dim=1)
    z = torch.sigmoid(W_z(z_input))

    r_1 = W_r(x).view(-1,1,hidden_size)
    r_2 = U_r(h_nei)
    r = torch.sigmoid(r_1 + r_2)
    
    gated_h = r * h_nei
    sum_gated_h = gated_h.sum(dim=1)
    h_input = torch.cat([x,sum_gated_h], dim=1)
    pre_h = torch.tanh(W_h(h_input))
    new_h = (1.0 - z) * sum_h + z * pre_h
    return new_h
def GRUs(x, h, W_z, W_r, U_r, W_h):
    #hidden_size = x.size()[-1]
    dim = x.dim()-1
    
    z_input = torch.cat([x,h],dim=dim)
    z = torch.sigmoid(W_z(z_input))
    r_1 = W_r(x).squeeze()
    r_2 = U_r(h)
    r = torch.sigmoid(r_1 + r_2)
    
    gated_h = torch.squeeze(r * h, dim)
    h_input = torch.cat([x,gated_h],dim=dim)
    pre_h = torch.tanh(W_h(h_input))
    new_h = (1.0 - z) * h + z * pre_h
    
    return new_h
def MPNN(fmess, mess_graph, W_g, depth, hidden_size):
    multi_layer_mess = []
    messages = torch.zeros(mess_graph.size(0), hidden_size).cuda()
    for i in range(depth):
        nei_message = index_select_ND(messages, 0, mess_graph)
        nei_message = nei_message.sum(dim=1)
        messages = torch.relu(W_g(torch.cat([fmess, nei_message], dim=1)))
        multi_layer_mess.append(messages)
        messages[0,:] = messages[0,:] * 0
        
    
    messages = torch.cat(multi_layer_mess, dim=1)
    messages[0,:] = messages[0,:] * 0 
    return messages