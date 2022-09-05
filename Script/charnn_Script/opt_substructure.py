'''
Author: 成凯阳
Date: 2022-03-15 14:34:41
LastEditors: 成凯阳
LastEditTime: 2022-06-20 19:02:18
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
from Dataset.get_Vocab import VocabDatasets,Vocabulary,collate_fn
from Model.model import CharRNN
import argparse
# from Utils.utils.metric import tonimato
from step5_optimzer_charnn import pretrain
from Utils.utils.train_utils import NoamLR,decrease_learning_rate
from Utils.utils.metric import fraction_valid_smiles

from torch import optim
import pandas as pd
from step5_optimzer_charnn import train_agents
import os


from Utils.utils.data_utils import fp_print,similarity
import heapq
from rdkit import rdBase
from rdkit.Chem import AllChem,MACCSkeys,Descriptors
from rdkit import DataStructs
rdBase.DisableLog('rdApp.error')

#CC(NC(=O)OC(C)(C)C)c1nc(CO)nn1Cc1ccccc1



def add_optimizer_args(parser):
    group = parser.add_argument_group("optimizering options")
    # group.add_argument("--restore_from", help="Checkpoint to load", type=bool, default=False)
    group.add_argument("--hidden", help="Model hidden size", type=int, default="256")
  
    # group.add_argument("--test_size", help="sample size", type=int, default="1000")
    group.add_argument("--num_layers", help="Model num layers", type=int, default="3")
    group.add_argument("--dropout", help="random sample point ", type=float, default="0.2")
    group.add_argument("--threshold", help="similarity threshold", type=float, default=0.7)

    group.add_argument("--lr", help="Learning rate", type=float, default=0.005)
    group.add_argument("--learn", help="Learning rate", type=float, default=0.001)
    group.add_argument("--beta1", help="Adam beta 1", type=float, default=0.9)
    group.add_argument("--beta2", help="Adam beta 2", type=float, default=0.998)
    group.add_argument("--eps", help="Adam epsilon", type=float, default=1e-9)
    group.add_argument("--weight_decay", help="Adam weight decay", type=float, default=1e-2)
    group.add_argument("--save_logs", help="path to save logs", type=str, default='/home/chengkaiyang/Main/opt/charnn/optmetrics.csv')
    group.add_argument("--gpu", help="Adam weight decay", type=str, default='True')
    group.add_argument("--max_length", help="max length to sample", type=int, default=100)
    group.add_argument("--n_batch", help="batch size to sample", type=int, default=250)
    group.add_argument("--save_logss", help="path to save logs", type=str, default='/home/chengkaiyang/Main/opt/charnn/metrics.csv')
    # group.add_argument("--train_p", help="Checkpoint to load", type=str, default='trainfilter.csv')
    # group.add_argument("--valid_p", help="Checkpoint to load", type=str, default='testfilter.csv')
    group.add_argument("--save_dir", help="path to save check", type=str, default='/home/chengkaiyang/Main/opt/charnn')
 
    group.add_argument("--restore_agent_from", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/s/model.2.pt')
    group.add_argument("--restore_prior_from", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/opt/charnn/Agent.ckpt')
    group.add_argument("--vocab_path", help="Vocab path to load", type=str, default='/home/chengkaiyang/Main/datanew/data/Voc.txt')
    # group.add_argument("--score_function", help="score function to choose", type=str, default='fingerprint')
    group.add_argument("--testmol", help="test mol similarity ", type=str, default='FC(F)(F)C1=CC(C(C)NC2=NC(C)=NC3=C2C=C(OC4COCC4)C(OC)=C3)=CC(N)=C1')
    group.add_argument("--cuda", help="use gpu device", type=str, default='cuda:0')
    group.add_argument("--s1", help="molecular substructure", type=str, default='FC(F)(F)C1=CC(C(C)N)=CC(N)=C1')
    group.add_argument("--s2", help="molecular substructure", type=str, default='OC1COCC1')
   #CC1CC(NBr)C=C1N=CC(=O)CN(C)O

  
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





def train_agent(query_fp,args,restore_prior_from=None,
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
    optimizer = torch.optim.Adam(Agent.parameters(), lr=args.learn)
    state = torch.load(restore_agent_from)
    pretrain_state_dict = state
    # pretrain_state_dict=state

    Agent.load_state_dict(pretrain_state_dict)
    df1 = pd.DataFrame(columns = ['smiles', 'best score','avg score'])
    df1.to_csv(args.save_logss)


    print("Model initialized, starting training...")
    raw_score=0
    raw_smi=args.testmol
    old=[]
    for step in range(n_steps):
        train_ss=[]
        val_ss=[]

        # Sample from Agent
        seqs = Agent.sample(args.n_batch)
        smiles=reback(Agent,seqs,args.n_batch)
     
        mols=[Chem.MolFromSmiles(i) for i in smiles]
        # p = Chem.MolFromSmiles('NC1=CC(C(F)(F)F)=CC(C)=C1')
        # p = Chem.MolFromSmiles('NC1=CC(C(F)(F)F)=CC(CC)=C1')
     
        # m=Chem.MolFromSmiles('C1CCOC1')
        p = Chem.MolFromSmiles(args.s1)
     
        # m=Chem.MolFromSmiles('C1CCOC1')
        m=Chem.MolFromSmiles(args.s2)
        subms =[x for x in mols if x.HasSubstructMatch(p) and x.HasSubstructMatch(m)]

        #subms =[x for x in mols if x.HasSubstructMatch(p) and x.HasSubstructMatch(m)]
        score=len(subms)
        smila=[Chem.MolToSmiles(i) for i in subms]
        # smiles=[]
        # smiles.extend(smila)
        # for i in range(1):
        #     # smiles.extend(smila)
        #     smiles.append(args.testmol)
        smila=list(set(smila))
        old_main_mol=smila
        for i in old_main_mol:
            if i not in old:
                old.append(i)
    
        smiles=[]
        smiles.extend(old)
        for i in range(1):
            # smiles.extend(smila)
            smiles.append(args.testmol)
            
            


 
 


       
       
        dict1 = {'smiles': smiles} 
        df = pd.DataFrame(dict1)  
 
        df.to_csv(args.save_logs,mode='w')
        if len(smila)>=int(args.n_batch//2):
            print('get terminal')
            break
        moldatatr=VocabDatasets(fname=args.save_logs,voc=voc)
        train_data = DataLoader(moldatatr, batch_size=1, shuffle=True, drop_last=True,
                      collate_fn=collate_fn)
        for i in tqdm(range(1), desc='Processing'):

            train_loss=pretrain(args,Prio=Agent,optimize=optimizer,train_dat=train_data,epoc=1)
          
            # with open(os.path.join(save_dir, "validsampled"), 'a') as f:
            #     f.write("{}\n".format(train_loss))
        lists = [score]
        data = pd.DataFrame([lists])
        data.to_csv(args.save_logss,mode='a',header=False,index=False)#
        torch.save(Agent.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))


   

if __name__ == "__main__":
    parser = argparse.ArgumentParser("preprocess and train")
    group=add_optimizer_args(parser)

    
    # parsing.add_train_args(parser)
    # parser.add_argument_group
    args = parser.parse_args()
    train_agents(query_fp=args.testmol,args=args,save_dir=args.save_dir,restore_agent_from=args.restore_agent_from,scoring_function_kwargs={})



    train_agent(query_fp=args.testmol,args=args,save_dir=args.save_dir,restore_agent_from=args.restore_prior_from,restore_prior_from=args.restore_prior_from,scoring_function_kwargs={})
