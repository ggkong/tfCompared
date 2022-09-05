'''
Author: 成凯阳
Date: 2022-03-15 14:34:41
LastEditors: 成凯阳
LastEditTime: 2022-06-20 19:08:53
FilePath: /Main/Script/charnn_Script/optimzer_charnn.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
#!/usr/bin/env python



from sklearn import metrics

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm
import numpy as np
# from Dataset.get_dataset import get_dataset,get_lookup_tables
from Dataset.get_Vocab import VocabDatasets,Vocabulary,collate_fn
from Model.model import CharRNN
import argparse

from Utils.utils.train_utils import NoamLR,decrease_learning_rate
from Utils.utils.metric import fraction_valid_smiles
from torch.nn import CrossEntropyLoss
from torch import optim
import pandas as pd

import os


from Utils.utils.data_utils import fp_print,similarity
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from Utils.utils.data_utils import Variable
from rdkit.Chem import MolFromSmiles
import heapq
from rdkit import rdBase
from rdkit.Chem import AllChem,MACCSkeys,Descriptors
from rdkit import DataStructs
rdBase.DisableLog('rdApp.error')

#CC(NC(=O)OC(C)(C)C)c1nc(CO)nn1Cc1ccccc1



def add_optimizer_args(parser):
    group = parser.add_argument_group("optimizering options")
 
    group.add_argument("--hidden", help="Model hidden size", type=int, default="256")
  
    # group.add_argument("--test_size", help="sample size", type=int, default="1000")
    group.add_argument("--num_layers", help="Model num layers", type=int, default="3")
    group.add_argument("--dropout", help="random sample point ", type=float, default="0.2")
  

    group.add_argument("--lr", help="Learning rate", type=float, default=0.0005)
    group.add_argument("--beta1", help="Adam beta 1", type=float, default=0.9)
    group.add_argument("--beta2", help="Adam beta 2", type=float, default=0.998)
    group.add_argument("--eps", help="Adam epsilon", type=float, default=1e-9)
    group.add_argument("--weight_decay", help="Adam weight decay", type=float, default=1e-2)
    group.add_argument("--save_logs", help="path to save logs", type=str, default='/home/chengkaiyang/Main/opt/charnn/optmetrics.csv')
    group.add_argument("--gpu", help="Adam weight decay", type=str, default='True')
    group.add_argument("--max_length", help="max length to sample", type=int, default=100)
    group.add_argument("--n_batch", help="batch size to sample", type=int, default=200)
    group.add_argument("--save_logss", help="path to save logs", type=str, default='/home/chengkaiyang/Main/opt/charnn/metrics.csv')
    # group.add_argument("--train_p", help="Checkpoint to load", type=str, default='trainfilter.csv')
    # group.add_argument("--valid_p", help="Checkpoint to load", type=str, default='testfilter.csv')
    group.add_argument("--save_dir", help="path to save check", type=str, default='/home/chengkaiyang/Main/opt/charnn')
 
    group.add_argument("--restore_agent_from", help="raw Checkpoint to load", type=str, default='/home/chengkaiyang/Main/s/model.2.pt')
    group.add_argument("--restore_prior_from", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/s/model.199.pt')
    group.add_argument("--vocab_path", help="Vocab path to load", type=str, default='/home/chengkaiyang/Main/datanew/data/Voc.txt')
  
    group.add_argument("--testmol", help="test mol similarity ", type=str, default='FC(F)(F)C1=CC(C(C)NC2=NC(C)=NC3=C2C=C(OC4COCC4)C(OC)=C3)=CC(N)=C1')
    group.add_argument("--cuda", help="use gpu device", type=str, default='cuda:0')
   #CC1CC(NBr)C=C1N=CC(=O)CN(C)O
    group.add_argument("--threshold", help="similarity threshold", type=float, default=0.7)
  
    return group
def reback(model,toens,n_batch):
    smis=[]
   

    for i in range(n_batch):
        stri=''
        for j in range(len(toens[i].cuda().cpu().numpy())):
            smi=model.vocabulary.reversed_vocab[int(toens[i][j])]
            stri+=smi
        if Chem.MolFromSmiles(stri):
            smis.append(stri)
        
    
       

        

    return smis  

def pretrain(args,Prio,optimize,train_dat,epoc):
    """Trains the Prior RNN"""


    losses = CrossEntropyLoss()


    Prio.train()
    total_loss=0

    for step, batch in tqdm(enumerate(train_dat), total=len(train_dat)):

        # Sample from DataLoader
        prevs,next,lens= batch
        device=args.cuda
        prevs=prevs.to(device)
        next=next.to(device)
        lens=[l.to(device)for l in lens]
     
        
        outputs, _, _ = Prio(prevs, lens)
        


        # Calculate loss
        loss = losses(outputs.view(-1, outputs.shape[-1]),
                            next.view(-1))

        # Calculate gradients and take a step
        optimize.zero_grad()
        loss.backward()
        optimize.step()
        total_loss+=loss.item()

    return total_loss/step

def train_agents(query_fp,args,
                restore_agent_from=None,
            
                scoring_function_kwargs=None,
                save_dir=None, learning_rate=0.001,
               n_steps=5000,
                
                ):

    voc = Vocabulary(init_from_file=args.vocab_path)



   
    Agent = CharRNN(voc,args)


    if args.gpu:
        device=args.cuda
       
        Agent=Agent.to(device)
    else:
     
        Agent=Agent
    optimizer = torch.optim.Adam(Agent.parameters(), lr=args.lr)
    state = torch.load(restore_agent_from)
    # pretrain_state_dict = state['state_dict']
    pretrain_state_dict=state['state_dict']

    Agent.load_state_dict(pretrain_state_dict)
    df1 = pd.DataFrame(columns = ['smiles', 'best score','avg score'])
    df1.to_csv(args.save_logss)

    print("Model initialized, starting training...")
    raw_score=0
    raw_smi=args.testmol

    for step in range(n_steps):
        train_ss=[]
        val_ss=[]

        # Sample from Agent
        seqs = Agent.sample(args.n_batch)
        smiles=reback(Agent,seqs,args.n_batch)
     
        # nseqs= Prior.sample(args.n_batch)
        # smiless=reback(Prior,nseqs,args.n_batch)
        score=fp_print(smiles,query_fp)
        # smiles=list(set(smiles))
        # idx = heapq.nlargest(200, range(len( score)),  score.__getitem__)
        # smiles=[smiles[id]  for id in idx]
        s=np.mean(score)
        if s>=args.threshold:
            print('get threshold')
            break
        max_score=score.index(max(score))
     
        max_smiles=smiles[max_score]
      
        idx=[i for i,a in enumerate(score) if a >s]
        smiles=[smiles[id]  for id in idx]

        for i in range(20):
            smiles.append(args.testmol)


       
       
        dict1 = {'smiles': smiles} 
        df = pd.DataFrame(dict1)  
 
        df.to_csv(args.save_logs,mode='w')
        moldatatr=VocabDatasets(fname=args.save_logs,voc=voc)
        train_data = DataLoader(moldatatr, batch_size=2, shuffle=True, drop_last=True,
                      collate_fn=collate_fn)
        for i in tqdm(range(1), desc='Processing'):

            train_loss=pretrain(args,Prio=Agent,optimize=optimizer,train_dat=train_data,epoc=1)
          
            # with open(os.path.join(save_dir, "validsampled"), 'a') as f:
            #     f.write("{}\n".format(train_loss))
        lists = [max_smiles,max(score), s]
        data = pd.DataFrame([lists])
        data.to_csv(args.save_logss,mode='a',header=False,index=False)#
        torch.save(Agent.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))
        raw_smi=max_smiles
        raw_score=max(score)

  
  
    # torch.save(Agent.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser("preprocess and train")
    group=add_optimizer_args(parser)

    
    # parsing.add_train_args(parser)
    # parser.add_argument_group
    args = parser.parse_args()



    train_agents(query_fp=args.testmol,args=args,save_dir=args.save_dir,restore_agent_from=args.restore_agent_from,scoring_function_kwargs={})
