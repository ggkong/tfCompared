'''
Author: 成凯阳
Date: 2022-03-20 07:39:17
LastEditors: 成凯阳
LastEditTime: 2022-06-17 01:37:26
FilePath: /Main/Script/aae_script/step3_aae_sample.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

import csv
from Dataset.get_dataset import get_dataset,get_lookup_tables
from Dataset.get_Vocab import VocabDatasets,Vocabulary
from Model.aae_model import AAE
import argparse


from Utils.utils.train_utils import NoamLR,decrease_learning_rate
from torch.nn import CrossEntropyLoss
from torch import optim
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from rdkit.Chem import MolFromSmiles, QED, AllChem
from rdkit.Chem.inchi import MolToInchiKey
from rdkit import DataStructs
from torch.nn.utils.rnn import pad_sequence

from Utils.utils.metric import canonic_smiles,compute_fragments,logP,SA,get_mol
# from utils import Variable, decrease_learning_rate
rdBase.DisableLog('rdApp.error')
def fraction_valid_smiles(smiles):
    """Takes a list of SMILES and returns fraction valid."""
    i = 0
    ko=[]
    mo=[]
    for smile in smiles:
        if Chem.MolFromSmiles(smile):
            i += 1
            ko.append(smile)
            mo.append(Chem.MolFromSmiles(smile))
    return i / len(smiles),ko ,mo
def add_sample_args(parser):
    group = parser.add_argument_group("Sampling options")
#     # file paths

    group.add_argument("--restore", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/savehuizong/aae/aaemodels.20.pt')
    group.add_argument('--latent_size', type=int, default=128,
                           help='Size of latent vectors')
    group.add_argument('--embedding_size', type=int, default=32,
                           help='Embedding size in encoder and decoder')
    group.add_argument('--decoder_hidden_size', type=int, default=512,
                           help='Size of hidden state for lstm '
                                'layers in decoder')
    group.add_argument('--encoder_bidirectional', type=bool, default=True,
                           help='If true to use bidirectional lstm '
                                'layers in encoder')
    group.add_argument('--discriminator_layers', nargs='+', type=int,
                           default=[640, 256],
                           help='Numbers of features for linear '
                                'layers in discriminator')
    group.add_argument('--encoder_hidden_size', type=int, default=512,
                           help='Size of hidden state for '
                                'lstm layers in encoder')
    group.add_argument('--discriminator_steps', type=int, default=1,
                           help='Discriminator training steps per one'
                                'autoencoder training step')
    group.add_argument('--decoder_num_layers', type=int, default=2,
                           help='Number of lstm layers in decoder')
    group.add_argument("--n_batch", help="return sample smiles size", type=int, default=100)
    group.add_argument("--generate_file", help="generate sample files", type=str, default='/home/chengkaiyang/Main/savehuizong/aae/aaeSample.txt')
    
  
 
 
    group.add_argument("--vocab_path", help="Vocab path to load", type=str, default='/home/chengkaiyang/Main/datanew/data/Voc.txt')
    group.add_argument("--hidden", help="Model hidden size", type=int, default="128")
    group.add_argument("--num_layers", help="Model layers", type=int, default="2")
    group.add_argument("--dropout", help="random sample point ", type=float, default="0.")
  
    group.add_argument("--gpu", help="use gpu or not", type=bool, default='True')
    group.add_argument("--cuda", help="use gpu device", type=str, default='cuda:0')

   
    return group



def reback(model,toens,n_batch):
    smis=[]
   

    for i in range(n_batch):
        stri=''
        for j in range(len(toens[i].cuda().cpu().numpy())):
            smi=model.vocabulary.reversed_vocab[int(toens[i][j])]
            stri+=smi
        stri=stri.split('<pad>')[0]
        smis.append(stri)
       

        

    return smis    



if __name__ == "__main__":
    parser = argparse.ArgumentParser("sample smiles")
    group=add_sample_args(parser)

    
    # parsing.add_train_args(parser)
    # parser.add_argument_group
   
    args = parser.parse_args()
    voc = Vocabulary(init_from_file=args.vocab_path)



    Prior = AAE(voc,args)
    if args.gpu:
        device=args.cuda
        Prior=Prior.to(device)
    else:
        Prior=Prior
 
    if args.restore=='':
        print("start new training ...")
        pass
    

 
    else:
        
      

        
        state = torch.load(args.restore)
        pretrain_state_dict = state['state_dict']
        Prior.load_state_dict(pretrain_state_dict)
        print("restore from {} ...".format(args.restore))


  

    new=Prior.sample(n_batch=args.n_batch,args=args)
    # new_pad=pad_sequence(new,batch_first=True)
    
    d=reback(Prior,new,args.n_batch)
   
    correct,smiles_list,mols=fraction_valid_smiles(d)
    # p = Chem.MolFromSmiles('NC1=CC(C(F)(F)F)=CC(CC)=C1')
     
    # m=Chem.MolFromSmiles('C1CCOC1')
    # p = Chem.MolFromSmiles('FC(F)(F)C1=CC(C(C)N)=CC(N)=C1')
     
    #     # m=Chem.MolFromSmiles('C1CCOC1')
    # m=Chem.MolFromSmiles('OC1COCC1')

    # subms =[x for x in mols if x.HasSubstructMatch(p) and x.HasSubstructMatch(m)]
    # d=[Chem.MolToSmiles(x) for x in subms]
    f=open(args.generate_file,"w")
 
    for line in d:
       f.write(line+'\n')