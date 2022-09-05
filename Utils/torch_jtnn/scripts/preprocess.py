import torch
import torch.nn as nn
from multiprocessing import Pool
import gc
import math, random, sys
from optparse import OptionParser
from argparse import ArgumentParser
import pickle as pickle

from torch_jtnn import *
import rdkit

def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree

class Preprocess():
    """Preprocess dataset for JTVAE

    Args:
        train_path (str):
        nsplits (int): Number of data set to split.
        njobs (int): Number of jobs for processing.
    """
    
    def __init__(self,train_path,nsplits,njobs):
        """[summary]
        """
        
        self.train_path = train_path
        self.nsplits = nsplits
        self.njobs = njobs

    def __call__(self):

        num_splits = self.nsplits
        njobs = self.njobs

        with open(self.train_path) as f:
            data = [line.strip("\r\n ").split()[0] for line in f]
        print("number of dataset: {}".format(len(data)))

        le = int((len(data) + num_splits - 1) / num_splits)

        for split_id in range(num_splits):
            print("Current {}".format(split_id))
            st = split_id * le
            with Pool(njobs) as pool: # The Pool object is created here to prevent memory overflow.
                sub_data = pool.map(tensorize, data[st : st + le])

            with open('tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
            del sub_data[:]
            gc.collect()

def preprocess(train_path,nsplits,njobs):
    """ Preprocess dataset for JTVAE

    Args:
        train_path (str):
        nsplits (int): Number of data set to split.
        njobs (int): Number of jobs for processing.
    """
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    proc = Preprocess(train_path,nsplits,njobs)
    proc()

def main():
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = ArgumentParser()
    parser.add_argument("-t", "--train", dest="train_path",type=str)
    parser.add_argument("-n", "--split", dest="nsplits", type=int,default=10)
    parser.add_argument("-j", "--jobs", dest="njobs", type=int,default=8)
    args = parser.parse_args()

    proc = Preprocess(**args.__dict__)
    proc()

if __name__ == "__main__":
    main()

