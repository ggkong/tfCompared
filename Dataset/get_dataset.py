'''
Author: 成凯阳
Date: 2022-03-11 07:46:31
LastEditors: 成凯阳
LastEditTime: 2022-03-12 17:13:21
FilePath: /Main/Dataset/get_dataset.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
from typing import Set
import pandas as pd
import os
import numpy as np
import re
import itertools
import csv
path='/opt/conda/envs/rdkit/lib/python3.7/site-packages/moses/dataset/data'
def get_lookup_tables(text):
    # chars = tuple(set(text))
    int2char = dict(enumerate(text))
    char2int = {ch: ii for ii, ch in int2char.items()}

# path = os.path.join(path,  split+'.csv.gz')


# smiles = pd.read_csv(path, compression='gzip')['SMILES'].values
# s=''
# for i in range(len(smiles)):
#     s+=smiles[i]
    

# print(smiles)
def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string
def get_dataset(split,path):
    path = os.path.join(path,  split+'.csv.gz')
    smiles = pd.read_csv(path, compression='gzip')['SMILES'].values
    text=''
    for i in range(len(smiles)):

        text+=smiles[i]
    int2char, char2int = get_lookup_tables(text)
    encoded = np.array([char2int[ch] for ch in text])
    chars = tuple(char2int.keys())
    return encoded,chars
    

# print(smiles)
def tokenize(smiles):
    """Takes a SMILES string and returns a list of tokens.
    This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
    '[xx]' as one token."""
    regex = '(\[[^\[\]]{1,6}\])'
    smiles = replace_halogen(smiles)
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
def get_dataset(split,path):
    path = os.path.join(path,  split+'.csv.gz')
    smiles = pd.read_csv(path, compression='gzip')['SMILES'].values
    
    np.savetxt('train.csv', smiles, fmt='%s')
if __name__ == "__main__":
    # get_dataset('train',path)
    smi='CC1C2CCC(C2)C1CN(CCO)C(=O)c1ccc(Cl)cc1'
   
    train_path='/user-data/Main/Data/train.csv'
    test_path='/user-data/Main/Data/testfilter.csv'
 
    with open(train_path,'rt') as csvfile:
        
        reader = csv.DictReader(csvfile)
        column = [tokenize(row['smiles']) for row in reader]
        result = list(itertools.chain(*column))
        result=set(result)
    with open(test_path,'rt') as csvfilet:
        
        readert = csv.DictReader(csvfilet)
        columnt = [tokenize(row['smiles']) for row in readert]
        resultt = list(itertools.chain(*columnt))
        resultt=set(resultt)
    res=resultt.union(result)
      
     
    with open('Data/Voc.txt', 'w') as f:
        for char in res:
            f.write(char + "\n")
 

 

     

    

    