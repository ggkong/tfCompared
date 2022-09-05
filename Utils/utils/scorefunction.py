'''
Author: 成凯阳
Date: 2022-03-15 14:35:39
LastEditors: 成凯阳
LastEditTime: 2022-06-20 03:21:38
FilePath: /Main/Utils/utils/scorefunction.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
#!/usr/bin/env python
from __future__ import print_function, division
from itertools import count
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem,MACCSkeys,Descriptors
from rdkit import DataStructs
from sklearn import svm
import time
import pickle
import re
import sys
import networkx as nx

from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdmolops
from rdkit import rdBase
# import selfies
import rdkit
import numpy as np
from rdkit import Chem
# from selfies import encoder, decoder
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem import Mol
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem import Crippen, MolFromSmiles, MolToSmiles
from rdkit.Chem import Descriptors

from Utils.utils.sascore import sascore

rdBase.DisableLog('rdApp.error')


# from https://github.com/gablg1/ORGAN/blob/master/organ/mol_metrics.py#L83
from collections import namedtuple
import math

from rdkit import Chem
from rdkit.Chem import MolSurf, Crippen
from rdkit.Chem import rdMolDescriptors as rdmd
from torch import log
from Utils.utils.sascore import sascore
rdBase.DisableLog('rdApp.error')
logP_mean = 2.457    # np.mean(logP_values)
logP_std = 1.434     # np.std(logP_values)
SA_mean = -3.053     # np.mean(SA_scores)
SA_std = 0.834       # np.std(SA_scores)
cycle_mean = -0.048  # np.mean(cycle_scores)
cycle_std = 0.287    # np.std(cycle_scores)
# def fp_print(smiles,query_fp):

#     score=[]
#     for s in smiles:
#         t=similarity(s,query_fp)
   
#         score.append(t)
     

   
#     return score
# def fp_pair_print(smiles):

#     score=[]
#     sc={}
#     for s in smiles:
#         t=logP_score(s)
#         # t=similarity(s,query_fp)
   
#         score.append(t)
#         sc[s]=t
     
def remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)
   
#     return score,sc
def single_fp(smile,query_fp):
    t=similarity(smile,query_fp)
    return t


def similarity(a, b):
    if a is None or b is None: 
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0



    fp1 = MACCSkeys.GenMACCSKeys(amol)
    fp2 = MACCSkeys.GenMACCSKeys(bmol)
    return DataStructs.FingerprintSimilarity(fp1, fp2)

def pair_log(smiles):
    ss=[]
    # count=0
    for s in smiles:
        if Chem.MolFromSmiles(s):
            mol=Chem.MolFromSmiles(s)
            log_p = Chem.Descriptors.MolLogP(mol)
            ss.append(log_p)
            # count+=1
        else:
            log_p=0
            ss.append(log_p)

        # mol=Chem.MolFromSmiles(s)
        # log_p = Chem.Descriptors.MolLogP(mol)
        # ss.append(log_p)
    return ss
def pair_log_sim(smiles,qf):
    ss=[]
    # count=0
    for s in smiles:
        if Chem.MolFromSmiles(s):
            mol=Chem.MolFromSmiles(s)
            log_p = Chem.Descriptors.MolLogP(mol)
            ss.append(log_p)
            # count+=1
        else:
            log_p=0
            ss.append(log_p)

        # mol=Chem.MolFromSmiles(s)
        # log_p = Chem.Descriptors.MolLogP(mol)
        # ss.append(log_p)
    return ss
def verify_sequence(smile):
    mol = Chem.MolFromSmiles(smile)
    return smile != '' and mol is not None and mol.GetNumAtoms() > 1


def randomize_smiles(mol):
    '''Returns a random (dearomatized) SMILES given an rdkit mol object of a molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol) :  RdKit mol object (None if invalid smile string smi)
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object  (None if invalid smile string smi)
    '''
    if not mol:
        return None

    Chem.Kekulize(mol)
    return rdkit.Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False,  kekuleSmiles=True)



def sanitize_smiles(smi):
    '''Return a canonical smile representation of smi
    
    Parameters:
    smi (string) : smile string to be canonicalized 
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful 
    '''
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)
    

def get_selfie_chars(selfie):
    '''Obtain a list of all selfie characters in string selfie
    
    Parameters: 
    selfie (string) : A selfie string - representing a molecule 
    
    Example: 
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    
    Returns:
    chars_selfie: list of selfie characters present in molecule selfie
    '''
    chars_selfie = [] # A list of all SELFIE sybols from string selfie
    while selfie != '':
        chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
        selfie = selfie[selfie.find(']')+1:]
    return chars_selfie

def batch_solubility(smiles, train_smiles=None):
    vals = [logP(s, train_smiles) if verify_sequence(s) else 0 for s in smiles]
    return vals


def logP(smile, train_smiles=None):
    try:
        low_logp = -2.12178879609
        high_logp = 6.0429063424
        logp = Crippen.MolLogP(Chem.MolFromSmiles(smile))
        val = remap(logp, low_logp, high_logp)
        val = np.clip(val, 0.0, 1.0)
        return val
    except ValueError:
        return 0.0


def mutate_selfie(selfie, max_molecules_len, write_fail_cases=False):
    '''Return a mutated selfie string (only one mutation on slefie is performed)
    
    Mutations are done until a valid molecule is obtained 
    Rules of mutation: With a 33.3% propbabily, either: 
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another
        3. Delete a random character
    
    Parameters:
    selfie            (string)  : SELFIE string to be mutated 
    max_molecules_len (int)     : Mutations of SELFIE string are allowed up to this length
    write_fail_cases  (bool)    : If true, failed mutations are recorded in "selfie_failure_cases.txt"
    
    Returns:
    selfie_mutated    (string)  : Mutated SELFIE string
    smiles_canon      (string)  : canonical smile of mutated SELFIE string
    '''
    valid=False
    fail_counter = 0
    chars_selfie = get_selfie_chars(selfie)
    
    while not valid:
        fail_counter += 1
                
        alphabet = list(selfies.get_semantic_robust_alphabet()) # 34 SELFIE characters 

        choice_ls = [1, 2, 3] # 1=Insert; 2=Replace; 3=Delete
        random_choice = np.random.choice(choice_ls, 1)[0]
        
        # Insert a character in a Random Location
        if random_choice == 1: 
            random_index = np.random.randint(len(chars_selfie)+1)
            random_character = np.random.choice(alphabet, size=1)[0]
            
            selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]

        # Replace a random character 
        elif random_choice == 2:                         
            random_index = np.random.randint(len(chars_selfie))
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                selfie_mutated_chars = [random_character] + chars_selfie[random_index+1:]
            else:
                selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index+1:]
                
        # Delete a random character
        elif random_choice == 3: 
            random_index = np.random.randint(len(chars_selfie))
            if random_index == 0:
                selfie_mutated_chars = chars_selfie[random_index+1:]
            else:
                selfie_mutated_chars = chars_selfie[:random_index] + chars_selfie[random_index+1:]
                
        else: 
            raise Exception('Invalid Operation trying to be performed')

        selfie_mutated = "".join(x for x in selfie_mutated_chars)
        sf = "".join(x for x in chars_selfie)
        
        try:
            smiles = decoder(selfie_mutated)
            mol, smiles_canon, done = sanitize_smiles(smiles)
            if len(selfie_mutated_chars) > max_molecules_len or smiles_canon=="":
                done = False
            if done:
                valid = True
            else:
                valid = False
        except:
            valid=False
            if fail_counter > 1 and write_fail_cases == True:
                f = open("selfie_failure_cases.txt", "a+")
                f.write('Tried to mutate SELFIE: '+str(sf)+' To Obtain: '+str(selfie_mutated) + '\n')
                f.close()
    
    return (selfie_mutated, smiles_canon)

def get_mutated_SELFIES(selfies_ls, num_mutations): 
    ''' Mutate all the SELFIES in 'selfies_ls' 'num_mutations' number of times. 
    
    Parameters:
    selfies_ls   (list)  : A list of SELFIES 
    num_mutations (int)  : number of mutations to perform on each SELFIES within 'selfies_ls'
    
    Returns:
    selfies_ls   (list)  : A list of mutated SELFIES
    
    '''
    for _ in range(num_mutations): 
        selfie_ls_mut_ls = []
        for str_ in selfies_ls: 
            
            str_chars = get_selfie_chars(str_)
            max_molecules_len = len(str_chars) + num_mutations
            
            selfie_mutated, _ = mutate_selfie(str_, max_molecules_len)
            selfie_ls_mut_ls.append(selfie_mutated)
            
        
        selfies_ls = selfie_ls_mut_ls.copy()
    return selfies_ls

def calc_score(smiles):
 
    if verify_sequence(smiles):
        try:
            molecule = MolFromSmiles(smiles)
            if Descriptors.MolWt(molecule) > 500:
                return 0
            current_log_P_value = Descriptors.MolLogP(molecule)
            current_SA_score = sascore.calculateScore(molecule)
            cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(molecule)))
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([len(j) for j in cycle_list])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6
            current_cycle_score = -cycle_length

            current_SA_score_normalized = (current_SA_score - SA_mean) / SA_std
            current_log_P_value_normalized = (current_log_P_value - logP_mean) / logP_std
            current_cycle_score_normalized = (current_cycle_score - cycle_mean) / cycle_std

            score = (current_SA_score_normalized
                     + current_log_P_value_normalized
                     + current_cycle_score_normalized)
            return score
        except Exception:
            return 0
    else:
        return 0





QEDproperties = namedtuple('QEDproperties', 'MW,ALOGP,HBA,HBD,PSA,ROTB,AROM,ALERTS')
ADSparameter = namedtuple('ADSparameter', 'A,B,C,D,E,F,DMAX')

WEIGHT_MAX = QEDproperties(0.50, 0.25, 0.00, 0.50, 0.00, 0.50, 0.25, 1.00)
WEIGHT_MEAN = QEDproperties(0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95)
WEIGHT_NONE = QEDproperties(1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00)

AliphaticRings = Chem.MolFromSmarts('[$([A;R][!a])]')

#
AcceptorSmarts = [
  '[oH0;X2]',
  '[OH1;X2;v2]',
  '[OH0;X2;v2]',
  '[OH0;X1;v2]',
  '[O-;X1]',
  '[SH0;X2;v2]',
  '[SH0;X1;v2]',
  '[S-;X1]',
  '[nH0;X2]',
  '[NH0;X1;v3]',
  '[$([N;+0;X3;v3]);!$(N[C,S]=O)]'
]
Acceptors = [Chem.MolFromSmarts(hba) for hba in AcceptorSmarts]

#
StructuralAlertSmarts = [
  '*1[O,S,N]*1',
  '[S,C](=[O,S])[F,Br,Cl,I]',
  '[CX4][Cl,Br,I]',
  '[#6]S(=O)(=O)O[#6]',
  '[$([CH]),$(CC)]#CC(=O)[#6]',
  '[$([CH]),$(CC)]#CC(=O)O[#6]',
  'n[OH]',
  '[$([CH]),$(CC)]#CS(=O)(=O)[#6]',
  'C=C(C=O)C=O',
  'n1c([F,Cl,Br,I])cccc1',
  '[CH1](=O)',
  '[#8][#8]',
  '[C;!R]=[N;!R]',
  '[N!R]=[N!R]',
  '[#6](=O)[#6](=O)',
  '[#16][#16]',
  '[#7][NH2]',
  'C(=O)N[NH2]',
  '[#6]=S',
  '[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]',
  'C1(=[O,N])C=CC(=[O,N])C=C1',
  'C1(=[O,N])C(=[O,N])C=CC=C1',
  'a21aa3a(aa1aaaa2)aaaa3',
  'a31a(a2a(aa1)aaaa2)aaaa3',
  'a1aa2a3a(a1)A=AA=A3=AA=A2',
  'c1cc([NH2])ccc1',
  '[Hg,Fe,As,Sb,Zn,Se,se,Te,B,Si,Na,Ca,Ge,Ag,Mg,K,Ba,Sr,Be,Ti,Mo,Mn,Ru,Pd,Ni,Cu,Au,Cd,' +
  'Al,Ga,Sn,Rh,Tl,Bi,Nb,Li,Pb,Hf,Ho]',
  'I',
  'OS(=O)(=O)[O-]',
  '[N+](=O)[O-]',
  'C(=O)N[OH]',
  'C1NC(=O)NC(=O)1',
  '[SH]',
  '[S-]',
  'c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]',
  'c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]',
  '[CR1]1[CR1][CR1][CR1][CR1][CR1][CR1]1',
  '[CR1]1[CR1][CR1]cc[CR1][CR1]1',
  '[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1',
  '[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1',
  '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1',
  '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1',
  'C#C',
  '[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]',
  '[$([N+R]),$([n+R]),$([N+]=C)][O-]',
  '[#6]=N[OH]',
  '[#6]=NOC=O',
  '[#6](=O)[CX4,CR0X3,O][#6](=O)',
  'c1ccc2c(c1)ccc(=O)o2',
  '[O+,o+,S+,s+]',
  'N=C=O',
  '[NX3,NX4][F,Cl,Br,I]',
  'c1ccccc1OC(=O)[#6]',
  '[CR0]=[CR0][CR0]=[CR0]',
  '[C+,c+,C-,c-]',
  'N=[N+]=[N-]',
  'C12C(NC(N1)=O)CSC2',
  'c1c([OH])c([OH,NH2,NH])ccc1',
  'P',
  '[N,O,S]C#N',
  'C=C=O',
  '[Si][F,Cl,Br,I]',
  '[SX2]O',
  '[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)',
  'O1CCCCC1OC2CCC3CCCCC3C2',
  'N=[CR0][N,n,O,S]',
  '[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2',
  'C=[C!r]C#N',
  '[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1',
  '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1',
  '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])',
  '[OH]c1ccc([OH,NH2,NH])cc1',
  'c1ccccc1OC(=O)O',
  '[SX2H0][N]',
  'c12ccccc1(SC(S)=N2)',
  'c12ccccc1(SC(=S)N2)',
  'c1nnnn1C=O',
  's1c(S)nnc1NC=O',
  'S1C=CSC1=S',
  'C(=O)Onnn',
  'OS(=O)(=O)C(F)(F)F',
  'N#CC[OH]',
  'N#CC(=O)',
  'S(=O)(=O)C#N',
  'N[CH2]C#N',
  'C1(=O)NCC1',
  'S(=O)(=O)[O-,OH]',
  'NC[F,Cl,Br,I]',
  'C=[C!r]O',
  '[NX2+0]=[O+0]',
  '[OR0,NR0][OR0,NR0]',
  'C(=O)O[C,H1].C(=O)O[C,H1].C(=O)O[C,H1]',
  '[CX2R0][NX3R0]',
  'c1ccccc1[C;!R]=[C;!R]c2ccccc2',
  '[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]',
  '[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]',
  '[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]',
  '[*]=[N+]=[*]',
  '[SX3](=O)[O-,OH]',
  'N#N',
  'F.F.F.F',
  '[R0;D2][R0;D2][R0;D2][R0;D2]',
  '[cR,CR]~C(=O)NC(=O)~[cR,CR]',
  'C=!@CC=[O,S]',
  '[#6,#8,#16][#6](=O)O[#6]',
  'c[C;R0](=[O,S])[#6]',
  'c[SX2][C;!R]',
  'C=C=C',
  'c1nc([F,Cl,Br,I,S])ncc1',
  'c1ncnc([F,Cl,Br,I,S])c1',
  'c1nc(c2c(n1)nc(n2)[F,Cl,Br,I])',
  '[#6]S(=O)(=O)c1ccc(cc1)F',
  '[15N]',
  '[13C]',
  '[18O]',
  '[34S]'
]

StructuralAlerts = [Chem.MolFromSmarts(smarts) for smarts in StructuralAlertSmarts]

adsParameters = {
  'MW': ADSparameter(A=2.817065973, B=392.5754953, C=290.7489764, D=2.419764353, E=49.22325677,
                     F=65.37051707, DMAX=104.9805561),
  'ALOGP': ADSparameter(A=3.172690585, B=137.8624751, C=2.534937431, D=4.581497897, E=0.822739154,
                        F=0.576295591, DMAX=131.3186604),
  'HBA': ADSparameter(A=2.948620388, B=160.4605972, C=3.615294657, D=4.435986202, E=0.290141953,
                      F=1.300669958, DMAX=148.7763046),
  'HBD': ADSparameter(A=1.618662227, B=1010.051101, C=0.985094388, D=0.000000001, E=0.713820843,
                      F=0.920922555, DMAX=258.1632616),
  'PSA': ADSparameter(A=1.876861559, B=125.2232657, C=62.90773554, D=87.83366614, E=12.01999824,
                      F=28.51324732, DMAX=104.5686167),
  'ROTB': ADSparameter(A=0.010000000, B=272.4121427, C=2.558379970, D=1.565547684, E=1.271567166,
                       F=2.758063707, DMAX=105.4420403),
  'AROM': ADSparameter(A=3.217788970, B=957.7374108, C=2.274627939, D=0.000000001, E=1.317690384,
                       F=0.375760881, DMAX=312.3372610),
  'ALERTS': ADSparameter(A=0.010000000, B=1199.094025, C=-0.09002883, D=0.000000001, E=0.185904477,
                         F=0.875193782, DMAX=417.7253140),
}


def ads(x, adsParameter):
  """ ADS function """
  p = adsParameter
  exp1 = 1 + math.exp(-1 * (x - p.C + p.D / 2) / p.E)
  exp2 = 1 + math.exp(-1 * (x - p.C - p.D / 2) / p.F)
  dx = p.A + p.B / exp1 * (1 - 1 / exp2)
  return dx / p.DMAX


def properties(mol):
  """
  Calculates the properties that are required to calculate the QED descriptor.
  """
  if mol is None:
    raise ValueError('You need to provide a mol argument.')
  mol = Chem.RemoveHs(mol)
  qedProperties = QEDproperties(
    MW=rdmd._CalcMolWt(mol),
    ALOGP=Crippen.MolLogP(mol),
    HBA=sum(len(mol.GetSubstructMatches(pattern)) for pattern in Acceptors
            if mol.HasSubstructMatch(pattern)),
    HBD=rdmd.CalcNumHBD(mol),
    PSA=MolSurf.TPSA(mol),
    ROTB=rdmd.CalcNumRotatableBonds(mol, rdmd.NumRotatableBondsOptions.Strict),
    AROM=Chem.GetSSSR(Chem.DeleteSubstructs(Chem.Mol(mol), AliphaticRings)),
    ALERTS=sum(1 for alert in StructuralAlerts if mol.HasSubstructMatch(alert)),
  )
  # The replacement
  # AROM=Lipinski.NumAromaticRings(mol),
  # is not identical. The expression above tends to count more rings
  # N1C2=CC=CC=C2SC3=C1C=CC4=C3C=CC=C4
  # OC1=C(O)C=C2C(=C1)OC3=CC(=O)C(=CC3=C2C4=CC=CC=C4)O
  # CC(C)C1=CC2=C(C)C=CC2=C(C)C=C1  uses 2, should be 0 ?
  return qedProperties


def qed(mol, w=WEIGHT_MEAN, qedProperties=None):
  """ Calculate the weighted sum of ADS mapped properties
  some examples from the QED paper, reference values from Peter G's original implementation
  >>> m = Chem.MolFromSmiles('N=C(CCSCc1csc(N=C(N)N)n1)NS(N)(=O)=O')
  >>> qed(m)
  0.253...
  >>> m = Chem.MolFromSmiles('CNC(=NCCSCc1nc[nH]c1C)NC#N')
  >>> qed(m)
  0.234...
  >>> m = Chem.MolFromSmiles('CCCCCNC(=N)NN=Cc1c[nH]c2ccc(CO)cc12')
  >>> qed(m)
  0.234...
  """
  if qedProperties is None:
      qedProperties = properties(mol)
  d = [ads(pi, adsParameters[name]) for name, pi in qedProperties._asdict().items()]
  t = sum(wi * math.log(di) for wi, di in zip(w, d))
  return math.exp(t / sum(w))


def weights_max(mol):
  """
  Calculates the QED descriptor using maximal descriptor weights.
  """
  return qed(mol, w=WEIGHT_MAX)


def weights_mean(mol):
  """
  Calculates the QED descriptor using average descriptor weights.
  """
  return qed(mol, w=WEIGHT_MEAN)


def weights_none(mol):
  """
  Calculates the QED descriptor using unit weights.
  """
  return qed(mol, w=WEIGHT_NONE)


def default(mol):
  """
  Calculates the QED descriptor using average descriptor weights.
  """
  return weights_mean(mol)


# ------------------------------------
#
#  doctest boilerplate
#
def _runDoctests(verbose=None):  # pragma: nocover
  import sys
  import doctest
  failed, _ = doctest.testmod(optionflags=doctest.ELLIPSIS, verbose=verbose)
  sys.exit(failed)


if __name__ == '__main__':  # pragma: nocover
  _runDoctests()



#   try:
#   	logp = Descriptors.MolLogP(m)
#   except:

#     print (m, Chem.MolToSmiles(m))
#     sys.exit('failed to make a molecule')

#   SA_score = -sascore.calculateScore(m)
#   #cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(m)))
#   cycle_list = m.GetRingInfo().AtomRings() #remove networkx dependence
#   if len(cycle_list) == 0:
#       cycle_length = 0
#   else:
#       cycle_length = max([ len(j) for j in cycle_list ])
#   if cycle_length <= 6:
#       cycle_length = 0
#   else:
#       cycle_length = cycle_length - 6
#   cycle_score = -cycle_length
#   #print cycle_score
#   #print SA_score
#   #print logp
#   logpmean=2.4713477640000012
#   LogP_std=1.4449891243959943
# #   SA_score_norm=(SA_score-SA_mean)/SA_std
#   logp_norm=(logp-logpmean)/LogP_std
#   cycle_score_norm=(cycle_score-cycle_mean)/cycle_std
#   score_one = SA_score_norm + logp_norm + cycle_score_norm
  

  