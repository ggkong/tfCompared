import sys, os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool
import numpy as np
from itertools import product
from joblib import Parallel, delayed
import json
from collections import defaultdict
from DeLinker_test import DenseGGNNChemModel
# from rdkit.Chem import rdmolops
from rdkit.DataStructs import TanimotoSimilarity


from Utils.utils.data_utils import read_file,preprocess,compute_distance_and_angle

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
def add_process_args(parser):
    group = parser.add_argument_group("processing options")

    group.add_argument("--scaff_2_path", help="scaff_1", type=str, default='*-C1CCOC1')
    group.add_argument("--scaff_1_path", help="scaff_2", type=str, default='OC1=C(F)C(-*)=CC=C1')
    group.add_argument("--n_batchs", help="final batch size to sample", type=int, default=200)
    # group.add_argument("--dataset", help="dataset", type=str, default='zinc')
    # group.add_argument("--freeze-graph-model", help="freeze-graph-model", type=bool, default=False)
    # group.add_argument("--restore", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/Script/DeLinker-master/models/pretrained_DeLinker_model.pickle')

    # group.add_argument("--generate_file", help="data to save", type=str, default='/home/chengkaiyang/Main/Script/DeLinker-master/1.txt')

  
    return group
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
# def return_csv(generated_all_smiles):
#     import pandas as pd
#     df1 = pd.DataFrame(columns = ['diversity',  'unique','valid'])
#     df1.to_csv('/home/chengkaiyang/Main/Script/DeLinker-master/datainfo.csv')
#     val=0
#     zong=len(generated_all_smiles)
#     valid_mol=[Chem.MolFromSmiles(i) for i in generated_all_smiles]
#     val_rate=valid_mol/zong
#     # for i in generated_all_smiles:
#     # if Chem.MolFromSmiles(i):
#     #     val+=1

#     fp_list = []
#     for molecule in valid_mol:
#         fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
#         fp_list.append(fp)
#     diversity = []
#     for i in range(len(fp_list)):
#         for j in range(i+1, len(fp_list)):
#             current_diverity  = 1 - float(TanimotoSimilarity(fp_list[i], fp_list[j]))
#             diversity.append(current_diverity)
#     div=np.mean(diversity)
#     unique_smiles = list(set(generated_all_smiles)) 
#     unique_ratio = len(unique_smiles)/len(generated_all_smiles)
#     list = [div,unique_ratio,val_rate ]
#     data = pd.DataFrame([list])
#     data.to_csv(args.save_log,mode='a',header=False,index=False)#
#     return valid_mol
if __name__ == "__main__":
    file_path=os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(file_path,'param.json'), 'r') as f:

        args= json.load(f)
    with open(os.path.join(file_path,'params.json'), 'r') as f:

        input_param= json.load(f)


    #train_agents(query_fp=args.testmol,args=args,save_dir=args.save_dir,restore_agent_from=args.restore_prior_from,scoring_function_kwargs={})

    args = argparse.Namespace(**args)
    
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
    a,b=mol_with_atom_index(combo_2d)
    print(a)
 
    mol_to_link=generate_conf(mol_to_link)
    dist, ang = compute_distance_and_angle(mol_to_link, "", Chem.MolToSmiles(mol_to_link))
    print(dist)
  

   
    with open(args.generate_file, 'w') as f:
        
        f.write("%s %s %s" % (Chem.MolToSmiles(mol_to_link), dist, ang))
    raw_data = read_file(args.generate_file)

    preprocess(raw_data, "zinc", "fragments_test", True)
    args=vars(args)
    # import json
    # json.dump(args,'/home/chengkaiyang/Main/Script/DeLinker-master')

    model = DenseGGNNChemModel(args,input_param)
    model.train()