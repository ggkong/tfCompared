import torch
import torch.nn as nn

import math, random, sys
import argparse
from torch_jtnn import *
import rdkit

class JTVAESampling():
    def __init__(self,**args):
        self.nsample = args['nsample']
        self.vocab = args['vocab']
        self.model = args['model']

        self.hidden_size = args.get('hidden_size',450)
        self.latent_size = args.get('lattent_size',56)
        self.depthT = args.get('depthT',20)
        self.depthG = args.get('depthG',3)

    def __call__(self):
        vocab,weights = zip(*[(x.split(',')[0].strip('\n\r'),int(x.split(',')[1].strip('\n\r'))) for x in open(self.vocab) ])
        vocab = Vocab(vocab,weights)

        model = JTNNVAE(vocab,self.hidden_size,self.latent_size,self.depthT,self.depthG)
        model.load_state_dict(torch.load(self.model))
        model = model.cuda()

        torch.manual_seed(0)

        result_smiles = []

        for i in range(self.nsample):
            result_smiles.append(model.sample_prior())
            print(result_smiles[i] )

        return result_smiles


    @staticmethod
    def chart_parser(parser=None):
        parser = parser or argparse.ArgumentParser()
        parser.add_argument('--nsample',type=int,required=True)
        parser.add_argument('--vocab',required=True)
        parser.add_argument('--model',required=True)

        parser.add_argument('--hidden_size',type=int,default=450)
        parser.add_argument('--latent_size',type=int,default=56)
        parser.add_argument('--depthT',type=int,default=20)
        parser.add_argument('--depthG',type=int,default=3)
        return parser

def jtvae_sampling(nsample,vocab,model,**kwargs):
    """sampling module 

    Args:
        nsample (int): Number of sampling
        vocab (str): Path of vocablary file
        model (str): Path of save directory

        hidden_size (int,optional): Size of hidden layer.
        latent_size (int,optional): Size of latent variable.
        depthT (int,optional): depth of tree.
        depthG (int,optional): depth of graph.

    """   
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    return JTVAESampling(nsample=nsample,vocab=vocab,model=model,**kwargs)()


def main():
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = JTVAESampling.chart_parser()

    args = parser.parse_args()

    JTVAESampling(**args.__dict__)()

if __name__ == "__main__":
    main()