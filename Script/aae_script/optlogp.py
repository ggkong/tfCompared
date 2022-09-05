'''
Descripttion: 
version: 
Author: 成凯阳
Date: 2022-05-07 08:31:38
LastEditors: 成凯阳
LastEditTime: 2022-06-20 18:38:20
'''


#!/usr/bin/env python





import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm
import numpy as np
from Dataset.get_dataset import get_dataset,get_lookup_tables
from Dataset.get_Vocab import AAEVocabDatasets,Vocabulary,collate_fnS
from Model.aae_model import AAE
import argparse
from step2_train_aae import pretrain

from Utils.utils.train_utils import NoamLR,decrease_learning_rate
from Utils.utils.metric import fraction_valid_smiles
from torch.nn import CrossEntropyLoss
from torch import log_, optim
import pandas as pd

import os
from rdkit import rdBase
from rdkit.Chem import AllChem,MACCSkeys,Descriptors
from rdkit import DataStructs

import re

from Utils.utils.scorefunction import pair_log



from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from Utils.utils.data_utils import Variable
from rdkit.Chem import MolFromSmiles

rdBase.DisableLog('rdApp.error')


def add_optimizer_args(parser):
    group = parser.add_argument_group("optimizering options")
    group.add_argument("--restore_from", help="Checkpoint to load", type=bool, default=False)
    group.add_argument("--hidden", help="Model hidden size", type=int, default="128")
  
   
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
  

    group.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    group.add_argument("--beta1", help="Adam beta 1", type=float, default=0.9)
    group.add_argument("--beta2", help="Adam beta 2", type=float, default=0.998)
    group.add_argument("--eps", help="Adam epsilon", type=float, default=1e-9)
    group.add_argument("--weight_decay", help="Adam weight decay", type=float, default=1e-4)
    group.add_argument("--save_logs", help="path to save logs", type=str, default='/home/chengkaiyang/Main/optdouble/aae/optmetrics.csv')
    group.add_argument("--gpu", help="use gpu", type= int, default='1')
    group.add_argument("--max_length", help="max length to sample", type=int, default=100)
    group.add_argument("--n_batch", help="batch size to sample", type=int, default=1000)
    group.add_argument("--num_layers", help="Model layers", type=int, default="2")
    # group.add_argument("--train_p", help="Checkpoint to load", type=str, default='trainfilter.csv')
    group.add_argument("--save_logss", help="path to save check", type=str, default='/home/chengkaiyang/Main/optdouble/aae/metric.csv')
    group.add_argument("--save_dir", help="path to save check", type=str, default='/home/chengkaiyang/Main/optdouble/aae/')
    group.add_argument("--dropout", help="random sample point ", type=float, default="0.")
    group.add_argument("--restore_agent_from", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/savehuizong/aae/aaemodels.20.pt')
    group.add_argument("--vocab_path", help="Vocab path to load", type=str, default='/home/chengkaiyang/Main/datanew/data/Voc.txt')
    group.add_argument("--score_function", help="score function to choose", type=str, default='fingerprint')
    group.add_argument("--testmol", help="test mol similarity ", type=str, default='CC(C)(C)OC(=O)N1CC[NH2+]CC1C(N)=O')
    group.add_argument("--cuda", help="use gpu device", type=str, default='cuda:0')
    group.add_argument("--sim", help="similarity", type=float, default=0.2)
    group.add_argument("--sthreshold", help="similarity threshold", type=float, default=0.7)
    group.add_argument("--lthreshold", help="logp threshold", type=float, default=3.5)
   

  
    return group


def reback(model,toens,n_batch):
    smis=[]
   

    for i in range(n_batch):
        stri=''
        for j in range(len(toens[i].cuda().cpu().numpy())):
            smi=model.vocabulary.reversed_vocab[int(toens[i][j])]
            stri+=smi
        stri=stri.split('<pad>')[0]
        if Chem.MolFromSmiles(stri):
            smis.append(stri)

        # smis.append(stri)
       

        

    return smis


def train_agent(query_fp,args,
                restore_agent_from=None,
                scoring_function=None,
                scoring_function_kwargs=None,
                save_dir=None, learning_rate=0.0005,
                batch_size=None, n_steps=70,
                num_processes=0, sigma=100,
                experience_replay=0):

    voc = Vocabulary(init_from_file=args.vocab_path)



   
    Agent = AAE(voc,args)


    if args.gpu:
        device=args.cuda
       
        Agent=Agent.to(device)
    else:
     
        Agent=Agent
        device='cpu'
        Agent=Agent.to(device)
    aoptimizer = optim.Adam(
          list(Agent.encoder.parameters()) +
                list(Agent.decoder.parameters()),
        lr=args.lr,
    
        weight_decay=args.weight_decay
    )
    # aoptimizer=torch.optim.RMSprop(list(Prior.encoder.parameters()) +
    #             list(Prior.decoder.parameters()), lr=0.05, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
 
    doptimizer = optim.Adam(
        Agent.discriminator.parameters(),
        lr=args.lr,
     
        weight_decay=args.weight_decay
    )
  
    state = torch.load(restore_agent_from,map_location=device)
    # pretrain_state_dict = state['state_dict']
    pretrain_state_dict=state['state_dict']

    Agent.load_state_dict(pretrain_state_dict)
    df1 = pd.DataFrame(columns = ['smiles', 'best score','avg score'])
    df1.to_csv(args.save_logss)

    print("Model initialized, starting training...")

    for step in range(n_steps):
        train_ss=[]
        val_ss=[]

        # Sample from Agent
        new=Agent.sample(n_batch=args.n_batch,args=args)
    # new_pad=pad_sequence(new,batch_first=True)
    
        smiles=reback(Agent,new,args.n_batch)
     
        score=pair_log(smiles)
        s=np.mean(score)
        if s>=args.lthreshold:
            print('get threshold')
            break
        max_score=score.index(max(score))
        max_smiles=smiles[max_score]
        idx=[i for i,a in enumerate(score) if a >s]
        # idx = heapq.nlargest(100, range(len( score)),  score.__getitem__)
        smiles=[smiles[id]  for id in idx]
      
       
       
        dict1 = {'smiles': smiles} 
        df = pd.DataFrame(dict1)  
 
        df.to_csv(args.save_logs,mode='w')
        moldatatr=AAEVocabDatasets(fname=args.save_logs,voc=voc)
     
  
        train_data = DataLoader(moldatatr, batch_size=10, shuffle=True, drop_last=True,
                      collate_fn=collate_fnS)
   
        for i in tqdm(range(1), desc='Processing'):
            train_loss,_,_=pretrain(args,Prior=Agent,autooptimizer=aoptimizer,disoptimizer=doptimizer,train_dat=train_data,epoc=i)

      
        lists = [max_smiles, max(score), s]
        data = pd.DataFrame([lists])
        data.to_csv(args.save_logss,mode='a',header=False,index=False)#
        torch.save(Agent.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))
   

  
  
    # torch.save(Agent.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser("preprocess and optimizer")
    group=add_optimizer_args(parser)

    

    args = parser.parse_args()



    train_agent(query_fp=args.testmol,args=args,save_dir=args.save_dir,restore_agent_from=args.restore_agent_from,scoring_function=args.score_function,scoring_function_kwargs={})
