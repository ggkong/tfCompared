'''
Author: 成凯阳
Date: 2022-03-24 01:43:54
LastEditors: 成凯阳
LastEditTime: 2022-06-18 05:10:40
FilePath: /Main/Script/charnn_Script/step1_get_Vocab.py

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
import csv
import itertools,argparse
from collections import Counter
def add_voc_args(parser):
    group = parser.add_argument_group("get vocab options")
#     # file paths
# 
    
  
 
    group.add_argument("--train_path", help="train dataset", type=str, default='/home/chengkaiyang/Main/datanew/data/train.csv')
    group.add_argument("--valid_path", help="valid dataset", type=str, default='/home/chengkaiyang/Main/datanew/data/val.csv')
    group.add_argument("--vocab_path", help="vocab dataset", type=str, default='/home/chengkaiyang/Main/datahuizong/Vocab.txt')
    return group
def tokenize(smiles):
    """Takes a SMILES string and returns a list of tokens.
    This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
    '[xx]' as one token."""
    regex = '(\[[^\[\]]{1,6}\])'
    # smiles = replace_halogen(smiles)
    char_list = re.split(regex, smiles)
    tokenized = []
    for char in char_list:
        if char.startswith('['):
            tokenized.append(char)
      
        else:
            chars = [unit for unit in char]
            [tokenized.append(unit) for unit in chars]
    # tokenized.append('EOS')
    return tokenized
if __name__ == "__main__":
    # get_dataset('train',path)
    parser = argparse.ArgumentParser("get vocab")
    group=add_voc_args(parser)

    
    # parsing.add_train_args(parser)
    # parser.add_argument_group
   
    args = parser.parse_args()
 
    with open(args.train_path,'rt') as csvfile:
        
        reader = csv.DictReader(csvfile)
        column = [tokenize(row['smiles']) for row in reader]
        result = list(itertools.chain(*column))
        result=set(result)
    with open(args.valid_path,'rt') as csvfilet:
        
        readert = csv.DictReader(csvfilet)
        columnt = [tokenize(row['smiles']) for row in readert]
        resultt = list(itertools.chain(*columnt))
        resultt=set(resultt)
    res=resultt.union(result)
    print(res)
      
     
    with open(args.vocab_path, 'w') as f:
        for char in result:
            f.write(char + "\n")
