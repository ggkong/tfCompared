'''
Author: 成凯阳
Date: 2022-03-12 19:39:27
LastEditors: 成凯阳
LastEditTime: 2022-06-20 18:59:51
FilePath: /Main/Script/charnn_Script/step3_sample_charnn.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
'''
Author: 成凯阳
Date: 2022-03-11 15:22:25
LastEditors: 成凯阳
LastEditTime: 2022-03-12 18:35:17
FilePath: /Main/Train/train.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
#!/usr/bin/env python
# import imp
# from lib2to3.pgen2 import token

# from matplotlib.pyplot import flag
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
import csv
from Dataset.get_Vocab import Vocabulary
from Model.model import CharRNN
import argparse
from torch import optim
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from rdkit.Chem import MolFromSmiles, QED, AllChem
from torch.nn.utils.rnn import pad_sequence
from Utils.utils.metric import canonic_smiles,compute_fragments,logP,SA,get_mol
# from utils import Variable, decrease_learning_rate
rdBase.DisableLog('rdApp.error')

def add_sample_args(parser):
    group = parser.add_argument_group("Sampling options")
#     # file paths
#     group.add_argument("--train_bin", help="Train npz", type=str, default="")
#     group.add_argument("--valid_bin", help="Valid npz", type=str, default="")
 
    group.add_argument("--restore", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/s/model.2.pt')
    group.add_argument("--max_length", help="max length to sample smiles", type=int, default=100)
    group.add_argument("--n_batch", help="return sample smiles size", type=int, default=100)
    # group.add_argument("--train_p", help="Checkpoint to load", type=str, default='trainfilter.csv')
    # group.add_argument("--valid_p", help="Checkpoint to load", type=str, default='testfilter.csv')
    group.add_argument("--hidden", help="Model hidden size", type=int, default="256")
    group.add_argument("--num_layers", help="Model num layers", type=int, default="3")
    group.add_argument("--dropout", help="random sample point ", type=float, default="0.2")

    group.add_argument("--save_dir", help="Checkpoint to save", type=str, default="/home/chengkaiyang/Main/savenew")
    group.add_argument("--lr", help="Learning rate", type=float, default=0.01)
    group.add_argument("--beta1", help="Adam beta 1", type=float, default=0.9)
    group.add_argument("--beta2", help="Adam beta 2", type=float, default=0.998)
    group.add_argument("--eps", help="Adam epsilon", type=float, default=1e-9)
    group.add_argument("--weight_decay", help="Adam weight decay", type=float, default=1e-2)
  
    group.add_argument("--gpu", help="use gpu or not", type=str, default='True')
    group.add_argument("--vocab_path", help="Vocab path to load", type=str, default='/home/chengkaiyang/Main/datanew/data/Voc.txt')
    group.add_argument("--generate_file", help="generate sample files", type=str, default='/home/chengkaiyang/Main/datanew/rSample.txt')
    group.add_argument("--cuda", help="use gpu device", type=str, default='cuda:0')
    return group



def reback(model,toens,n_batch):
    smis=[]
   

    for i in range(n_batch):
        stri=''
        for j in range(len(toens[i].cuda().cpu().numpy())):
            smi=model.vocabulary.reversed_vocab[int(toens[i][j])]
            stri+=smi
        
        smis.append(stri)
       

        

    return smis    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sample smiles")
    group=add_sample_args(parser)

    
    # parsing.add_train_args(parser)
    # parser.add_argument_group
   
    args = parser.parse_args()
    voc = Vocabulary(init_from_file=args.vocab_path)

    # df1 = pd.DataFrame(columns = ['epoch', 'step', 'loss'])
    # df1.to_csv("/user-data/Main/logs/train_metrics.csv")
    # df2 = pd.DataFrame(columns = ['epoch', 'step', 'loss'])
    # df2.to_csv("/user-data/Main/logs/valid_metrics.csv")
    

    # Create a Dataset from a SMILES file
    # path1='/user-data/Main/Data/'
    # train_path=os.path.join(path1,args.train_p)
    # test_path=os.path.join(path1,args.valid_p)
    # moldatatr=VocabDatasets(fname=train_path,voc=voc)
    # moldatate=VocabDatasets(fname=test_path,voc=voc)
 
    # train_data = DataLoader(moldatatr, batch_size=16384, shuffle=True, drop_last=True,
    #                   collate_fn=collate_fn)
    # valid_data = DataLoader(moldatate, batch_size=4096, shuffle=True, drop_last=True,
    #                   collate_fn=collate_fn)

    Prior = CharRNN(voc,args)
    if args.gpu:
        device=args.cuda
        Prior=Prior.to(device=device)
    else:
        Prior=Prior
    optimizer = optim.AdamW(
        Prior.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    if args.restore=='':
        print("start new training ...")
        pass
    

 
    else:
        
      

        
        state = torch.load(args.restore)
        pretrain_state_dict = state["state_dict"]
        Prior.load_state_dict(pretrain_state_dict)
        print("restore from {} ...".format(args.restore))


    # state = torch.load(args.restore)
    # pretrain_state_dict = state["state_dict"]
    # Prior.load_state_dict(pretrain_state_dict)

    new=Prior.sample(n_batch=args.n_batch,max_length=args.max_length)
    new_pad=pad_sequence(new,batch_first=True)
    
    d=reback(Prior,new,args.n_batch)
    correct,smiles_list,mols=fraction_valid_smiles(d)
    print(correct)
     
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
    # prior_likelihood, _ = Prior.likehood(new_pad)


    


       
  

 

    



  
