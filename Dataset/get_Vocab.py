'''
Author: 成凯阳
Date: 2022-03-11 09:35:27
LastEditors: 成凯阳
LastEditTime: 2022-06-23 00:57:47
FilePath: /Main/Dataset/get_Vocab.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
from base64 import encode
import re
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import pickle
import numpy as np
from torch.autograd import Variable
from torch.utils import data
# from dgl.data.utils import save_graphs,load_graphs
from collections import Counter
from rdkit import Chem
import itertools


class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""
    def __init__(self, init_from_file=None, max_length=140):
        self.special_tokens = ['<bos>','<eos>','<pad>']
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        
        
        if init_from_file: self.init_from_file(init_from_file)

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.int64)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        smiles_matrix = torch.tensor(smiles_matrix, dtype=torch.long,
                              device=self.device
                              )

        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.vocab['<eos>']: break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def tokenize(self, smiles):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = '(\[[^\[\]]{1,6}\])'
        smiles=smiles.split(',')[1]
        # smiles = smiles.replace("Cl", "L").replace("R", "Br")
      
        char_list = re.split(regex, smiles)
        tokenized = []
        tokenized.append(self.special_tokens[0])
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        tokenized.append(self.special_tokens[1])
        # tokenized.append('EOS')
        return tokenized

    def add_characters(self, chars):
        """Adds characters to the vocabulary"""
        for char in chars:
            self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = [self.special_tokens[0]]+char_list + [self.special_tokens[1]]+[self.special_tokens[2]]
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.vectors = torch.eye(len(self.vocab))

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
        self.add_characters(chars)
        self.device=torch.device( "cpu")
        # self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)


class CharDataset(Dataset):

    def __init__(self, data, content,block_size):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
    
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    # def __len__(self):
    #     return math.ceil(len(self.data) / (self.block_size + 1))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        len_smiles = len(smiles)
        dix =  [self.stoi[s] for s in smiles]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
class OPTCharDataset(Dataset):

    def __init__(self, data, content,block_size):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
    
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        smiles = self.data[idx]
        len_smiles = len(smiles)
        dix =  [self.stoi[s] for s in smiles]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y       
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out



class VocabDatasets(Dataset):
    def __init__(self, fname, voc):
        self.voc = voc
        self.smiles = []
        with open(fname, 'r') as f:
            for line in f:
                self.smiles.append(line[:-1])
            self.smiles=self.smiles[1:]
            # if fname=='/home/chengkaiyang/Main/datanew/data/424.csv':
            #     self.smiles=self.smiles[1:]
            # else:
            #     self.smiles=self.smiles[1:20000]

            

    def __getitem__(self, item):
        
        pred = self.voc.encode(self.voc.tokenize(self.smiles[item]))
        pre=pred[:-1]
        next=pred[1:]
        length=len(pred) - 1
        length=torch.tensor(length,
                                   dtype=torch.long,
                                   device=torch.device( "cpu"))
       
        # return self.smiles[item], self.graphs[item], self.rxns[item]
        # pre=
        return Variable(pre),Variable(next),Variable(length),self.voc.vocab['<pad>']

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))
def collate_fn(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param data: 元组，第一个元素：句子序列数据，第二个元素：长度 第2维：句子标签
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
    # (tensor([1, 2, 3, 5]), 4, 0)
    # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
    # pack_padded_sequence函数要求要按照序列的长度倒序排列
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    data_length = [xi[2] for xi in batch_data]
    sent_seq = [xi[0] for xi in batch_data]
    label = [xi[1] for xi in batch_data]
    pa=[xi[3] for xi in batch_data]
    pre = pad_sequence(sent_seq, batch_first=True, padding_value=pa[0])
    next = pad_sequence(label, batch_first=True, padding_value=pa[0])
    return  pre,next,data_length
class GPTVocabDatasets(Dataset):
    def __init__(self, fname, voc):
        self.voc = voc
        self.smiles = []
        with open(fname, 'r') as f:
            for line in f:
                self.smiles.append(line[:-1])
            self.smiles=self.smiles[1:]
            # if fname=='/home/chengkaiyang/Main/datanew/data/train.csv':
                
            #     self.smiles=self.smiles[1:380000]
            # else:
            #     self.smiles=self.smiles[1:250000]

    def __getitem__(self, item):
        
        pred = self.voc.encode(self.voc.tokenize(self.smiles[item]))
        pre=pred[:-1]
        next=pred[1:]
        length=len(pred) - 1
        pred=torch.tensor(pred,
                                   dtype=torch.long)
       
     
        return Variable(pred),self.voc.vocab['<pad>']

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))
def collate_fnnGP(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param data: 元组，第一个元素：句子序列数据，第二个元素：长度 第2维：句子标签
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
    # (tensor([1, 2, 3, 5]), 4, 0)
    # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
    # pack_padded_sequence函数要求要按照序列的长度倒序排列
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    # data_length = [xi[2] for xi in batch_data]
    sent_seq = [xi[0] for xi in batch_data]
    # label = [xi[1] for xi in batch_data]
    pa=[xi[1] for xi in batch_data]
    # sent_seqs = [xi[0][1:] for xi in batch_data]
 
    pred = pad_sequence(sent_seq, batch_first=True, padding_value=pa[0])
    pre=pred[:,:-1]
    nexts=pred[:,1:]
    # nexts = pad_sequence(sent_seqs, batch_first=True, padding_value=pa[0])
 
    # next = pad_sequence(label, batch_first=True, padding_value=pa[0])
    return  pre,nexts
class VAEVocabDatasets(Dataset):
    def __init__(self, fname, voc):
        self.voc = voc
        self.smiles = []
        with open(fname, 'r') as f:
            for line in f:
                self.smiles.append(line[:-1])
            # if fname=='/home/chengkaiyang/Main/datanew/data/train.csv':
                
            #     self.smiles=self.smiles[1:380000]
            # else:
            self.smiles=self.smiles[1:]

    def __getitem__(self, item):
        
        pred = self.voc.encode(self.voc.tokenize(self.smiles[item]))
        pre=pred[:-1]
        next=pred[1:]
        length=len(pred) - 1
        pred=torch.tensor(pred,
                                   dtype=torch.long)
       
     
        return Variable(pred),self.voc.vocab['<pad>']

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))
def collate_fnn(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param data: 元组，第一个元素：句子序列数据，第二个元素：长度 第2维：句子标签
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
    # (tensor([1, 2, 3, 5]), 4, 0)
    # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
    # pack_padded_sequence函数要求要按照序列的长度倒序排列
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    # data_length = [xi[2] for xi in batch_data]
    sent_seq = [xi[0] for xi in batch_data]
    # label = [xi[1] for xi in batch_data]
    pa=[xi[1] for xi in batch_data]
    pre = pad_sequence(sent_seq, batch_first=True, padding_value=pa[0])
    # next = pad_sequence(label, batch_first=True, padding_value=pa[0])
    return  pre
class GPTVocabDatasets(Dataset):
    def __init__(self, fname, voc):
        self.voc = voc
        self.smiles = []
        with open(fname, 'r') as f:
            for line in f:
                self.smiles.append(line[:-1])
            if fname=='/home/chengkaiyang/Main/datanew/data/train.csv':
                
                self.smiles=self.smiles[1:380000]
            else:
                self.smiles=self.smiles[1:10000]

    def __getitem__(self, item):
        
        pred = self.voc.encode(self.voc.tokenize(self.smiles[item]))
        pre=pred[:-1]
        next=pred[1:]
        length=len(pred) - 1
        pred=torch.tensor(pred,
                                   dtype=torch.long)
       
     
        return Variable(pre),self.voc.vocab['<pad>'],Variable(next)

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))
def collate_fnnGP(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param data: 元组，第一个元素：句子序列数据，第二个元素：长度 第2维：句子标签
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
    # (tensor([1, 2, 3, 5]), 4, 0)
    # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
    # pack_padded_sequence函数要求要按照序列的长度倒序排列
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    # data_length = [xi[2] for xi in batch_data]
    sent_seq = [xi[0] for xi in batch_data]
    # label = [xi[1] for xi in batch_data]
    pa=[xi[1] for xi in batch_data]
    sent_seqs = [xi[2] for xi in batch_data]
    pre = pad_sequence(sent_seq, batch_first=True, padding_value=pa[0])
    next=pad_sequence(sent_seqs, batch_first=True, padding_value=pa[0])
    # next = pad_sequence(label, batch_first=True, padding_value=pa[0])
    return  pre,next
class AAEVocabDatasets(Dataset):
    def __init__(self, fname, voc):
        self.voc = voc
        self.smiles = []
        with open(fname, 'r') as f:
            for line in f:
                self.smiles.append(line[:-1])
            self.smiles=self.smiles[1:]
            # if fname=='/home/chengkaiyang/Main/datanew/data/train.csv':
                
            #     self.smiles=self.smiles[1:250000]
            # else:
            #     self.smiles=self.smiles[1:10000]

    def __getitem__(self, item):
        
        pred = self.voc.encode(self.voc.tokenize(self.smiles[item]))
        pre=pred[:-1]
        next=pred[1:]
        length=len(pred) - 1
        length=torch.tensor(length,
                                   dtype=torch.long,
                                   device=torch.device("cpu"))
       
        # return self.smiles[item], self.graphs[item], self.rxns[item]
        # pre=
        return Variable(pre),Variable(next),Variable(length),self.voc.vocab['<pad>'],Variable(pred)

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))
def collate_fnS(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param data: 元组，第一个元素：句子序列数据，第二个元素：长度 第2维：句子标签
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
    # (tensor([1, 2, 3, 5]), 4, 0)
    # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
    # pack_padded_sequence函数要求要按照序列的长度倒序排列
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    data_length = [xi[2] for xi in batch_data]
    len_shr=[xi[2]-1 for xi in batch_data]
    sent_seq = [xi[0] for xi in batch_data]
    label = [xi[1] for xi in batch_data]
    pa=[xi[3] for xi in batch_data]
    padd=[xi[4] for xi in batch_data]
    pred = pad_sequence(padd, batch_first=True, padding_value=pa[0])
    pre = pad_sequence(sent_seq, batch_first=True, padding_value=pa[0])
    next = pad_sequence(label, batch_first=True, padding_value=pa[0])
    # return  pre,next,data_length
    return (pred, len_shr), (pre, data_length),(next, data_length)

class OPTAAEVocabDatasets(Dataset):
    def __init__(self, fname, voc):
        self.voc = voc
        self.smiles = fname
     

    def __getitem__(self, item):
        self.smiles[item]='0,'+self.smiles[item]
        
        pred = self.voc.encode(self.voc.tokenize(self.smiles[item]))
        pre=pred[:-1]
        next=pred[1:]
        length=len(pred) - 1
        length=torch.tensor(length,
                                   dtype=torch.long,
                                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
       
        # return self.smiles[item], self.graphs[item], self.rxns[item]
        # pre=
        return Variable(pre),Variable(next),Variable(length),self.voc.vocab['<pad>'],Variable(pred)

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))
# class GVAEVocabDatasets(Dataset):
#     def __init__(self, fname, voc):
#         self.voc = voc
#         self.smiles = []
#         with open(fname, 'r') as f:
#             for line in f:
#                 self.smiles.append(line[:-1])
#             if fname=='/home/chengkaiyang/Main/Data/trainfilter.csv':
                
#                 self.smiles=self.smiles[1:600000]
#             else:
#                 self.smiles=self.smiles[1:20000]

#     def __getitem__(self, item):
#         atom_types = ['C', 'N', 'O', 'S', 'F',  'P', 'Cl', 'Br','I']
        
#         pred = self.voc.encode(self.voc.tokenize(self.smiles[item]))
#         mol=Chem.MolFromSmiles(self.smiles[item].split(',')[1])
#         pre=pred[:-1]
#         next=pred[1:]
#         length=len(pred) - 1
#         pred=torch.tensor(pred,
#                                    dtype=torch.long)
   
#         bond_featurizer = torch.tensor(get_bond_features(mol),
#                                    dtype=torch.long)
#         atom_featurizer =  torch.tensor(get_atom_features(mol),
#                                    dtype=torch.long)
#         product_adj = torch.tensor(Chem.rdmolops.GetAdjacencyMatrix(mol),
#                                    dtype=torch.long)
       
     
#         return Variable(pred),self.voc.vocab['<pad>'],Variable(bond_featurizer),Variable(atom_featurizer),Variable(product_adj)

#     def __len__(self):
#         return len(self.smiles)

#     def __str__(self):
#         return "Dataset containing {} structures.".format(len(self))
# def collate_fnnsnS(batch_data):
#     """
#     自定义 batch 内各个数据条目的组织方式
#     :param data: 元组，第一个元素：句子序列数据，第二个元素：长度 第2维：句子标签
#     :return: 填充后的句子列表、实际长度的列表、以及label列表
#     """
#     # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
#     # (tensor([1, 2, 3, 5]), 4, 0)
#     # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
#     # pack_padded_sequence函数要求要按照序列的长度倒序排列
#     batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
#     # data_length = [xi[2] for xi in batch_data]
#     sent_seq = [xi[0] for xi in batch_data]
#     # label = [xi[1] for xi in batch_data]
#     pa=[xi[1] for xi in batch_data]
#     bf=[xi[2] for xi in batch_data]
#     af=[xi[3] for xi in batch_data]
#     ad=[xi[4] for xi in batch_data]
#     pre = pad_sequence(sent_seq, batch_first=True, padding_value=pa[0])
#     bf = torch.stack(bf,dim = 0)
#     af = torch.stack(af,dim = 0)
#     ad = torch.stack(ad,dim = 0)

#     # next = pad_sequence(label, batch_first=True, padding_value=pa[0])
#     return  pre,bf,af,ad

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    path='/home/chengkaiyang/Main/Data/Voc.txt'
    voc=Vocabulary(init_from_file=path)
    print(voc)
    # fname='/home/chengkaiyang/Main/molgandata/val'
    # train_data = MOlVAETRVocabDatasets(fname)
    
    # mydataloader = DataLoader(train_data,
    #                               batch_size=4,
    #                               shuffle=True,
    #                               num_workers=0,
    #                               collate_fn=collate)
    # for i,x in enumerate(mydataloader):

    #     (a,b,c)=x
    #     mols=[]
    #     # for i in a:
    #     #     i=int(i)
    #     print(a)
            # f=open(os.path.join(fname,'rxn_data_{}.pkl'.format(i)),'rb')
            
            # reaction_data = pickle.load(f)
            # mol=reaction_data['mols'] 
            # mols.append(mol)
         

       
        
    # path1='/user-data/Main/Data/1.csv'
    # data=VocabDatasets(fname=path1,voc=voc)
    # print(data)
    # c=data.voc.tokenize(data.smiles[1])
    
    # print(data.voc.encode(c))
    # train_da=DataLoader(data,batch_size=4,shuffle=False,num_workers=0,collate_fn=collate_fn)
    # for a,b,c in train_da:
    #     print(a,b,c)
    