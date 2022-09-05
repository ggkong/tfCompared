
'''
Author: 成凯阳
Date: 2022-03-11 15:22:25
LastEditors: 成凯阳
LastEditTime: 2022-06-20 18:42:03
FilePath: /Main/Train/train.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
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

import csv
from rdkit.DataStructs import TanimotoSimilarity
import argparse
from collections import Counter
from multiprocessing import Pool
from Utils.utils.train_utils import NoamLR,decrease_learning_rate
from torch.nn import CrossEntropyLoss
from torch import device, optim
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from rdkit.Chem import MolFromSmiles, QED, AllChem
from rdkit.Chem.inchi import MolToInchiKey
from rdkit import DataStructs
from rdkit.Chem import AllChem,MACCSkeys
from Utils.fcd_torch.fcd_torch.fcd import FCD
from torch.nn.utils.rnn import pad_sequence
from Utils.utils.metric import mapper,tonimatos,avesimilarity
from Utils.utils.metric import  fragmenter, weight,canonic_smiles,compute_fragments,logP,SA,get_mol,fraction_valid_smiles,cos_similarity,compute_fragments,fingerprints,average_agg_tanimoto,novelty
from numpy import *
from Utils.utils.data_utils import fp_print
rdBase.DisableLog('rdApp.error')

def add_optimizer_args(parser):
    group = parser.add_argument_group("test molecular")
   
    group.add_argument("--FragMetric", help="use FragMetric", type=bool, default=True)
    group.add_argument("--SNNMetric", help="use SNNMetric", type=bool, default=True)
    group.add_argument("--WassersteinMetric", help="use WassersteinMetric", type=bool, default=True)
    group.add_argument("--fractionMetric", help="use fractionMetric", type=bool, default=True)
    group.add_argument("--fcdMetric", help="use fcdMetric", type=bool, default=True)
    group.add_argument("--diversity", help="use diversity", type=bool, default=True)
    group.add_argument("--unique", help="use unique", type=bool, default=True)
    group.add_argument("--nolverty", help="use nolverty", type=bool, default=True)
    group.add_argument("--sim", help="use similarity", type=bool, default=True)

    group.add_argument("--gpu", help="use gpu or not", type=str, default='True')
    group.add_argument("--s1", help="molecular substructure", type=str, default='NC1=CC(C(F)(F)F)=CC(CC)=C1')
    group.add_argument("--s2", help="molecular substructure", type=str, default='C1CCOC1')
    group.add_argument("--max_length", help="max length to smiles sequence", type=int, default=100)
    group.add_argument("--n_batch", help="The number of batch sample", type=int, default=128)
    group.add_argument("--n_job", help="num worker to compute", type=int, default=1)
    group.add_argument("--has_sub", help="whether molecular has both substructure", type=bool, default=True)
  
    group.add_argument("--valid_p", help="test dataset", type=str, default='/home/chengkaiyang/Main/datanew/data/debu.csv')
  #CC(C)(C)OC(=O)N1CC[NH2+]CC1C(N)=O
    group.add_argument("--generate_file", help="file to generate", type=str, default='/home/chengkaiyang/Main/1.txt')
    group.add_argument("--testmol", help="test mol similarity ", type=str, default='FC(F)(F)C1=CC(N)=CC(C(NC2=NC=NC3=C2C=C(OC4COCC4)C(O)=C3)C)=C1')
    group.add_argument("--cuda", help="use gpu device", type=str, default='cuda:0')

  
    return group
   



if __name__ == "__main__":
    parser = argparse.ArgumentParser("test molecular")
    group=add_optimizer_args(parser)
  

    
    # parsing.add_train_args(parser)
    # parser.add_argument_group
    args = parser.parse_args()
    devices=args.cuda
 

  
    test_smi=[]
    dff=pd.read_csv(args.valid_p)
    rochange=dff['smiles'].tolist()
    with open(args.valid_p,'r',encoding='utf-8') as csvfile:

        reader = csv.reader(csvfile)
        rows = [row[1] for row in reader]
    
        rochange=rows[1:]
    # rochange=[Chem.MolToSmiles(row) for row in rochange]
    f = open(args.generate_file,"r") 
    lines = f.readlines() 




    if args.fractionMetric:
        correct,smiles_list,mols=fraction_valid_smiles(lines)
        unique_smiles = list(set(smiles_list)) 
        unique_ratio = len(unique_smiles)/len(smiles_list)
        print(f'Mean Valid: {correct:.2f}')
        print("Mean Unique of molecules % = {}".format(np.mean(unique_ratio)))
    if args.sim:

        x=fp_print(smiles_list,args.testmol)
        b = mean(x)
        print(f'Mean similarity: {b:.2f}')
    if args.has_sub:
        p = Chem.MolFromSmiles(args.s1)
        m=Chem.MolFromSmiles(args.s2)
        subms =[x for x in mols if x.HasSubstructMatch(p) and x.HasSubstructMatch(m)]
        unique_ratios = len(subms)/len(smiles_list)
        print("has of molecules % = {}".format(np.mean(unique_ratios)))



    if args.FragMetric:
        # x_fp=tonimatos(smile_list=smiles_list)
       
        # y_fp=tonimatos(smile_list=rochange)
      
        # h=avesimilarity(x_fp,y_fp)
        h=average_agg_tanimoto(fingerprints(smiles_list,n_jobs=args.n_job), fingerprints(rochange,n_jobs=args.n_job),device=devices)

      
        print(f'Mean FragMetric: {h:.2f}')
    if args.diversity:
        fp_list = []
        for molecule in mols:

            fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
            fp_list.append(fp)
        diversity = []
        for i in range(len(fp_list)):
            for j in range(i+1, len(fp_list)):
                current_diverity  = 1 - float(TanimotoSimilarity(fp_list[i], fp_list[j]))
                diversity.append(current_diverity)
        print("Diversity of molecules % = {}".format(np.mean(diversity)))
    if args.nolverty:
        nol=novelty(smiles_list,rochange)
        print(f'Mean nolverty: {nol:.2f}')


    if args.fcdMetric:

        fcd = FCD(device=args.cuda, n_jobs=32)
  
        hh=fcd(smiles_list,rochange)
       
        # hh=fcd(smiles_list, rochange)
        print(f'Mean fcdMetric: {hh:.2f}')
    if args.WassersteinMetric:

        qed = [QED.qed(mol) for mol in mols]
        qed = np.array(qed)
        logp = [logP(mol) for mol in mols]
        log=np.array(logp)
        SA_score = [SA(mol) for mol in mols]
        SA_score=np.array(SA_score)
        we_score = [weight(mol) for mol in mols]
        we_score=np.array(we_score)
        print(f'Mean QED: {np.mean(qed):.2f}')
     
        print(f'Mean logp: {np.mean(log):.2f}')
        print(f'Mean SA: {np.mean(SA_score):.2f}')
        print(f'Mean weight: {np.mean(we_score):.2f}')
    if args.SNNMetric:
        sss=compute_fragments(smiles_list,rows=rochange)
      
        
        
       



        
        print(f'Mean SNNMetric: {sss:.2f}')


     
   


 

    


       
  

 

    



  
