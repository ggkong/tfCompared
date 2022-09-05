'''
Descripttion: 
version: 
Author: 成凯阳
Date: 2022-08-29 04:49:14
LastEditors: 成凯阳
LastEditTime: 2022-08-31 13:13:39
'''
# from ast import pattern
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from scaffold_constrained_model import scaffold_constrained_RNN
from data_structs import Vocabulary, Experience
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique
import json
from rdkit.Chem import QED
from Utils.utils.metric import  logP,SA
import argparse
import torch
from rdkit.Chem import PandasTools, QED, Descriptors, rdMolDescriptors
import numpy as np
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

def  youhua(mol,save):
    
    mol = AllChem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.MolToMolFile(mol,save)
def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    tmp_mol = Chem.Mol(mol)
    for idx in range(atoms):
        tmp_mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(tmp_mol.GetAtomWithIdx(idx).GetIdx()))
    return Chem.MolToSmiles(tmp_mol),tmp_mol

def mol_with_atom_indexs(mol):
   
 
    for atom in mol.GetAtoms():
        if atom.GetSymbol() =='*' and atom.GetIdx()!=0:
            index=atom.GetIdx()
            print(atom.GetIdx())
            break
    return index
  
    # return mol,Chem.MolToSmiles(mol)
def return_csv(generated_all_smiles,path):
    import pandas as pd
    df1 = pd.DataFrame(columns = ['diversity',  'unique','valid'])
    df1.to_csv(path)
    val=0
    zong=len(generated_all_smiles)
    valid_mol=[Chem.MolFromSmiles(i) for i in generated_all_smiles]
    val_rate=len(valid_mol)/zong
    # for i in generated_all_smiles:
    # if Chem.MolFromSmiles(i):
    #     val+=1

    fp_list = []
    for molecule in valid_mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
        fp_list.append(fp)
    diversity = []
    for i in range(len(fp_list)):
        for j in range(i+1, len(fp_list)):
            current_diverity  = 1 - float(TanimotoSimilarity(fp_list[i], fp_list[j]))
            diversity.append(current_diverity)
    div=np.mean(diversity)
    unique_smiles = set(generated_all_smiles)
    unique_ratio = len(unique_smiles)/len(generated_all_smiles)
    list = [div,unique_ratio,val_rate ]
    d = {'diversity':[div],  'unique':[unique_ratio],'valid':[val_rate]}
    data = pd.DataFrame(data=d, columns = ['diversity',  'unique','valid'],dtype=float)


  
    data.to_csv(path,header=True,index=False)#
    return valid_mol
def concat(frag_1,frag_2):

    combo_no_exit = Chem.CombineMols(frag_1,frag_2)
    # combo = Chem.CombineMols(combo_no_exit, Chem.MolFromSmiles("*.*"))
    combo_2d = Chem.Mol(combo_no_exit)
    _ = AllChem.Compute2DCoords(combo_2d)
    edcombo = Chem.EditableMol(combo_2d)
    mol_to_link = edcombo.GetMol()
    Chem.SanitizeMol(mol_to_link)
    return combo_2d,combo_no_exit,mol_to_link
def generate_conf(mol_to_link):

    AllChem.EmbedMolecule(mol_to_link)
    AllChem.MMFFOptimizeMolecule(mol_to_link)
    return mol_to_link
def cal_mol_props(smi, verbose=False):
    try:
        m=Chem.MolFromSmiles(smi)
        if not m:
            return None
        mw = np.round(Descriptors.MolWt(m),1)
        logp = np.round(Descriptors.MolLogP(m),2)
        hbd = rdMolDescriptors.CalcNumLipinskiHBD(m)
        hba = rdMolDescriptors.CalcNumLipinskiHBA(m)
        psa = np.round(Descriptors.TPSA(m),1)
        rob= rdMolDescriptors.CalcNumRotatableBonds(m)
        qed= np.round(QED.qed(m),2)
        chiral_center=len(Chem.FindMolChiralCenters(m,includeUnassigned=True))
        if verbose:
            print ('Mw ',mw)
            print ('Logp ',logp)
            print ('HBD ', hbd)
            print ('HBA ', hba)
            print ('TPSA ', psa)
            print ('RotB ', rob)
            print ('QED ', qed)
            print ('chiral_center ', chiral_center)
        return mw,logp,hbd,hba,psa,rob,qed,chiral_center
    
    except Exception as e:
        print (e)
        return None
if __name__ == "__main__":

    file_path=os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(file_path,'param.json'), 'r') as f:

        args= json.load(f)
    with open(os.path.join(file_path,'params.json'), 'r') as f:

        input_param= json.load(f)


    #train_agents(query_fp=args.testmol,args=args,save_dir=args.save_dir,restore_agent_from=args.restore_prior_from,scoring_function_kwargs={})

    args = argparse.Namespace(**args)
    flag1=input_param['scaff_1_path'].count('@') > 0
    if flag1:

        m = Chem.MolFromSmiles(input_param['scaff_1_path'])
        input_param['scaff_1_path']=Chem.MolToSmiles(m, isomericSmiles=False)
    flag2=input_param['scaff_2_path'].count('@') > 0
    if flag2:

        m = Chem.MolFromSmiles(input_param['scaff_2_path'])
        input_param['scaff_2_path']=Chem.MolToSmiles(m, isomericSmiles=False)
    
    molraw1=Chem.MolFromSmiles(input_param['scaff_1_path'])
    # youhua(molraw1,args.xing_1_path)
    molraw2=Chem.MolFromSmiles(input_param['scaff_2_path'])
    if molraw1 is None:
        print("the first linker is None")
        sys.exit(0)
    if molraw2 is None:
        print("the second linker is None")
        sys.exit(0)
    if input_param['scaff_1_path'].find('*')==-1:
        print("the first linker is invalid")
        sys.exit(0)
    if input_param['scaff_2_path'].find('*')==-1:
        print("the second linker is invalid")
        sys.exit(0)
    # try:
    #     molraw1=molraw11
    # except:
    #     print("the first linker is None")
    # try:
    #     molraw2=molraw22
    # except:
    #     print("the first linker is None")





    combo_2d,combo_no_exit,mol_to_link=concat(molraw1,molraw2)
    # a,b=mol_with_atom_index(combo_2d)
    # print(a)
 
    mol_to_link=generate_conf(mol_to_link)
    ms=Chem.MolToSmiles(mol_to_link)
    mol_to_link=Chem.MolFromSmiles(ms)
    index=mol_with_atom_indexs(mol_to_link)
    mol_to_link = Chem.RWMol(mol_to_link)
    mol_to_link.AddBond(index,0,Chem.BondType.SINGLE)
    patterns=Chem.MolToSmiles(mol_to_link)
    patterns=patterns.replace('**', '*')
    # patterns='Cc1c(C#N)cccc1C(C)*N1CCOCC1'
    voc = Vocabulary(init_from_file=args.voc)
    
    Agent = scaffold_constrained_RNN(voc)
    print(Agent.rnn.state_dict)
    if torch.cuda.is_available():
        Agent.rnn.load_state_dict(torch.load(args.pretrain_path))
    else:
        Agent.rnn.load_state_dict(torch.load(args.pretrain_path, map_location=lambda storage, loc: storage))
    seqs, agent_likelihood, entropy = Agent.sample(pattern=patterns, batch_size=input_param['n_batchs'])
    core_1 = Chem.MolFromSmiles('CCc1cccc(C#N)c1C')
    core_2 = Chem.MolFromSmiles('N1CCOCC1')

    smiles = seq_to_smiles(seqs, voc)
    import pandas as pd
    df2 = pd.DataFrame(columns = ["smiles","qed","sa","logp","mw","hbd","hba","rob"])
    df2.to_csv('task2.csv',mode='a',header=["smiles","qed","sa","logp","mw","hbd","hba","rob"],index=False)#
    generated_all_smiles = []
    idx=0
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m:
            if flag1 or flag2:
                dec_isomers = list(EnumerateStereoisomers(m))
                dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(
                mol, isomericSmiles=True)) for mol in dec_isomers]
                smiles_3d = [Chem.MolToSmiles(mol, isomericSmiles=True)for mol in dec_isomers]
                
                mw,logp,hbd,hba,psa,rob,qed,chiral_center=cal_mol_props(s, verbose=False)
                # lista = [smiles_3d[0],QED.qed(mols),SA(mols),logP(mols)]
                lista = [smiles_3d[0],qed,SA(m),logp,mw,hbd,hba,rob]
                datass = pd.DataFrame([lista])
                    
                datass.to_csv('task2.csv',mode='a',header=False,index=False)#
                generated_all_smiles.append(s)
                data_di=args.tmp_dir
                filepath=os.path.join(data_di, '{}.png'.format(idx))
                Draw.MolToFile(Chem.MolFromSmiles(smiles_3d[0]), filepath, size=(400,400))
                idx+=1
            else:
                          
                
                mw,logp,hbd,hba,psa,rob,qed,chiral_center=cal_mol_props(s, verbose=False)
                # lista = [smiles_3d[0],QED.qed(mols),SA(mols),logP(mols)]
                lista = [s,qed,SA(m),logp,mw,hbd,hba,rob]
                datass = pd.DataFrame([lista])
                    
                datass.to_csv('task2.csv',mode='a',header=False,index=False)#
                generated_all_smiles.append(s)
                data_di=args.tmp_dir
                filepath=os.path.join(data_di, '{}.png'.format(idx))
                Draw.MolToFile(Chem.MolFromSmiles(s), filepath, size=(400,400))
                idx+=1

    vali=return_csv(generated_all_smiles,'datas_h.csv')
