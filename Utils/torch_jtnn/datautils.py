import torch
from torch.utils.data import Dataset, DataLoader
from Utils.torch_jtnn.mol_tree import MolTree,MolPropTree
import numpy as np
from Utils.torch_jtnn.jtnn_enc import JTNNEncoder
from Utils.torch_jtnn.mpn import MPN
from Utils.torch_jtnn.jtmpn import JTMPN
import _pickle as pickle
import os, random

class PairTreeFolder(object):

    def __init__(self, path, vocab, avocab, batch_size, num_workers=0, shuffle=True, y_assm=True, replicate=None, add_target=False):
        self.path = path
        self.batch_size = batch_size
        self.vocab = vocab
        self.avocab = avocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle
        self.add_target = add_target

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        number_of_files = 0
        for fn in os.listdir(self.path):
            if not fn.endswith("pkl"): continue
            
            number_of_files += 1
            fn = os.path.join(self.path, fn)
            import sys
            sys.path.append('/home/chengkaiyang/Main/Modof/model/')
            with open(fn, 'rb') as f:
                data = pickle.load(f)
                
            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch
            
            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()
            
            dataset = PairTreeDataset(batches, self.vocab, self.avocab, self.y_assm, add_target=self.add_target)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])
            
            for b in dataloader:
                yield b
            del data, batches, dataset, dataloader
        
        if number_of_files == 0: raise ValueError("The names of data files must end with 'pkl'. " + \
                                            "No such file exist in the train path")

class MolTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            import sys
            # sys.path.append('/home/chengkaiyang/Main/Utils/fast_jtnn')
            with open(fn, 'rb') as f:
                data = pickle.load(f)

            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = MolTreeDataset(batches, self.vocab, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])#, num_workers=self.num_workers)

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, avocab, y_assm, add_target):
        self.data = data
        self.vocab = vocab
        self.avocab = avocab
        self.y_assm = y_assm
        self.add_target = add_target

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        batch_data = self.data[idx]
        
        tree1_batch = [dpair[0] for dpair in batch_data]
        tree2_batch = [dpair[1] for dpair in batch_data]
        
        
        x_batch = MolPropTree.tensorize(tree1_batch, self.vocab, self.avocab, target=False, add_target=self.add_target)
        y_batch = MolPropTree.tensorize(tree2_batch, self.vocab, self.avocab, target=True, add_target=self.add_target)
        
        return x_batch, y_batch, tree1_batch, tree2_batch

class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, assm=self.assm)

def tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder,mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            #Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder,batch_idx)

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1
