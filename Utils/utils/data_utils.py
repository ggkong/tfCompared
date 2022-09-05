'''
Author: 成凯阳
Date: 2022-03-12 20:31:04
LastEditors: 成凯阳
LastEditTime: 2022-06-19 12:48:22
FilePath: /Main/Utils/utils/data_utils.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
import random
import torch
import numpy as np
import torch.nn.functional as F
import copy
import torch,json
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import AllChem,MACCSkeys,Descriptors
from rdkit import DataStructs
from itertools import chain, product
from rdkit.Chem import rdmolops
SMALL_NUMBER = 1e-7
LARGE_NUMBER= 1e10

geometry_numbers=[3, 4, 5, 6] # triangle, square, pentagen, hexagon

# bond mapping
bond_dict = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, "AROMATIC": 3}
number_to_bond= {0: Chem.rdchem.BondType.SINGLE, 1:Chem.rdchem.BondType.DOUBLE, 
                 2: Chem.rdchem.BondType.TRIPLE, 3:Chem.rdchem.BondType.AROMATIC}

def dataset_info(dataset): #qm9, zinc, cep
    if dataset=='qm9':
        return { 'atom_types': ["H", "C", "N", "O", "F"],
                 'maximum_valence': {0: 1, 1: 4, 2: 3, 3: 2, 4: 1},
                 'number_to_atom': {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"},
                 'bucket_sizes': np.array(list(range(4, 28, 2)) + [29])
               }
    elif dataset=='zinc':
        return { 'atom_types': ['Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 'I1(0)',
                'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)', 'O2(0)', 'S2(0)','S4(0)', 'S6(0)'],
                 'maximum_valence': {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5:1, 6:2, 7:3, 8:4, 9:1, 10:2, 11:2, 12:4, 13:6, 14:3},
                 'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5:'I', 6:'N', 7:'N', 8:'N', 9:'O', 10:'O', 11:'S', 12:'S', 13:'S'},
                 'bucket_sizes': np.array([28,31,33,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,53,55,58,84]) 
               }
    
    elif dataset=="cep":
        return { 'atom_types': ["C", "S", "N", "O", "Se", "Si"],
                 'maximum_valence': {0: 4, 1: 2, 2: 3, 3: 2, 4: 2, 5: 4},
                 'number_to_atom': {0: "C", 1: "S", 2: "N", 3: "O", 4: "Se", 5: "Si"},
                 'bucket_sizes': np.array([25,28,29,30, 32, 33,34,35,36,37,38,39,43,46])
               }
    else:
        print("the datasets in use are qm9|zinc|cep")
        exit(1)
def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)
def need_kekulize(mol):
    for bond in mol.GetBonds():
        if bond_dict[str(bond.GetBondType())] >= 3:
            return True
    return False
def onehot(idx, len):
    z = [0 for _ in range(len)]
    z[idx] = 1
    return z
def fp_print(smiles,query_fp):

    score=[]
    for s in smiles:
        t=similarity(s,query_fp)
   
        score.append(t)
     

   
    return score
def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    num_lines = len(lines)
    data = []
    for i, line in enumerate(lines):
        toks = line.strip().split(' ')
        if len(toks) == 3:
            smi_frags, abs_dist, angle = toks
            smi_mol = smi_frags
            smi_linker = ''
        elif len(toks) == 5:
            smi_mol, smi_linker, smi_frags, abs_dist, angle = toks
        else:
            print("Incorrect input format. Please check the README for useage.")
            exit()
        data.append({'smi_mol': smi_mol, 'smi_linker': smi_linker, 
                     'smi_frags': smi_frags,
                     'abs_dist': [abs_dist,angle]})
        if i % 2000 == 0:
            print('Finished reading: %d / %d' % (i, num_lines), end='\r')
    print('Finished reading: %d / %d' % (num_lines, num_lines))
    return data
def align_mol_to_frags(smi_molecule, smi_linker, smi_frags):
    try:
        # Load SMILES as molecules
        mol = Chem.MolFromSmiles(smi_molecule)
        frags = Chem.MolFromSmiles(smi_frags)
        linker = Chem.MolFromSmiles(smi_linker)
        # Include dummy atoms in query
        du = Chem.MolFromSmiles('*')
        qp = Chem.AdjustQueryParameters()
        qp.makeDummiesQueries=True
    
        # Renumber molecule based on frags (incl. dummy atoms)
        aligned_mols = []

        sub_idx = []
        # Get matches to fragments and linker
        qfrag = Chem.AdjustQueryProperties(frags,qp)
        frags_matches = list(mol.GetSubstructMatches(qfrag, uniquify=False))
        qlinker = Chem.AdjustQueryProperties(linker,qp)
        linker_matches = list(mol.GetSubstructMatches(qlinker, uniquify=False))

        # Loop over matches
        for frag_match, linker_match in product(frags_matches, linker_matches):
            # Check if match
            f_match = [idx for num, idx in enumerate(frag_match) if frags.GetAtomWithIdx(num).GetAtomicNum() != 0]
            l_match = [idx for num, idx in enumerate(linker_match) if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 and idx not in f_match]
            # If perfect match, break
            if len(set(list(f_match)+list(l_match))) == mol.GetNumHeavyAtoms():
                break
        # Add frag indices
        sub_idx += frag_match
        # Add linker indices to end
        sub_idx += [idx for num, idx in enumerate(linker_match) if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 and idx not in sub_idx]

        aligned_mols.append(Chem.rdmolops.RenumberAtoms(mol, sub_idx))
        aligned_mols.append(frags)

        nodes_to_keep = [i for i in range(len(frag_match))]
        
        # Renumber dummy atoms to end
        dummy_idx = []
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_idx.append(atom.GetIdx())
        for i, mol in enumerate(aligned_mols):
            sub_idx = list(range(aligned_mols[1].GetNumHeavyAtoms()+2))
            for idx in dummy_idx:
                sub_idx.remove(idx)
                sub_idx.append(idx)
            if i == 0:
                mol_range = list(range(mol.GetNumHeavyAtoms()))
            else:
                mol_range = list(range(mol.GetNumHeavyAtoms()+2))
            idx_to_add = list(set(mol_range).difference(set(sub_idx)))
            sub_idx.extend(idx_to_add)
            aligned_mols[i] = Chem.rdmolops.RenumberAtoms(mol, sub_idx)

        # Get exit vectors
        exit_vectors = []
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                if atom.GetIdx() in nodes_to_keep:
                    nodes_to_keep.remove(atom.GetIdx())
                for nei in atom.GetNeighbors():
                    exit_vectors.append(nei.GetIdx())

        if len(exit_vectors) != 2:
            print("Incorrect number of exit vectors")

        return (aligned_mols[0], aligned_mols[1]), nodes_to_keep, exit_vectors

    except:
        print("Could not align")
        return ([],[]), [], []
def to_graph_mol(mol, dataset):
    if mol is None:
        return [], []
    # Kekulize it
    if need_kekulize(mol):
        rdmolops.Kekulize(mol)
        if mol is None:
            return None, None
    # remove stereo information, such as inward and outward edges
    Chem.RemoveStereochemistry(mol)

    edges = []
    nodes = []
    for bond in mol.GetBonds():
        if mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() == 0 or mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() == 0:
            continue
        else:
            edges.append((bond.GetBeginAtomIdx(), bond_dict[str(bond.GetBondType())], bond.GetEndAtomIdx()))
            assert bond_dict[str(bond.GetBondType())] != 3
    for atom in mol.GetAtoms():
        if dataset=='qm9' or dataset=="cep":
            nodes.append(onehot(dataset_info(dataset)['atom_types'].index(atom.GetSymbol()), len(dataset_info(dataset)['atom_types'])))
        elif dataset=='zinc': # transform using "<atom_symbol><valence>(<charge>)"  notation
            symbol = atom.GetSymbol()
            valence = atom.GetTotalValence()
            charge = atom.GetFormalCharge()
            atom_str = "%s%i(%i)" % (symbol, valence, charge)

            if atom_str not in dataset_info(dataset)['atom_types']:
                if "*" in atom_str:
                    continue
                else:
                    print('unrecognized atom type %s' % atom_str)
                    return [], []

            nodes.append(onehot(dataset_info(dataset)['atom_types'].index(atom_str), len(dataset_info(dataset)['atom_types'])))

    return nodes, edges
def preprocess(raw_data, dataset, name, test=False):
    print('Parsing smiles as graphs.')
    processed_data =[]
    total = len(raw_data)
    for i, (smi_mol, smi_frags, smi_link, abs_dist) in enumerate([(mol['smi_mol'], mol['smi_frags'], 
                                                                   mol['smi_linker'], mol['abs_dist']) for mol in raw_data]):
        if test:
            smi_mol = smi_frags
            smi_link = ''
        (mol_out, mol_in), nodes_to_keep, exit_points = align_mol_to_frags(smi_mol, smi_link, smi_frags)
        if mol_out == []:
            continue
        nodes_in, edges_in = to_graph_mol(mol_in, dataset)
        nodes_out, edges_out = to_graph_mol(mol_out, dataset)
        if min(len(edges_in), len(edges_out)) <= 0:
            continue
        processed_data.append({
                'graph_in': edges_in,
                'graph_out': edges_out, 
                'node_features_in': nodes_in,
                'node_features_out': nodes_out, 
                'smiles_out': smi_mol,
                'smiles_in': smi_frags,
                'v_to_keep': nodes_to_keep,
                'exit_points': exit_points,
                'abs_dist': abs_dist
            })
        if i % 500 == 0:
            print('Processed: %d / %d' % (i, total), end='\r')
    print('Processed: %d / %d' % (total, total))
    print('Saving data')
    with open('molecules_%s.json' % name, 'w') as f:
        json.dump(processed_data, f)
    print('Length raw data: \t%d' % total)
    print('Length processed data: \t%d' % len(processed_data))
##### Structural information #####
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)
def compute_distance_and_angle(mol, smi_linker, smi_frags):
    try:
        frags = [Chem.MolFromSmiles(frag) for frag in smi_frags.split(".")]
        frags = Chem.MolFromSmiles(smi_frags)
        linker = Chem.MolFromSmiles(smi_linker)
        # Include dummy in query
        du = Chem.MolFromSmiles('*')
        qp = Chem.AdjustQueryParameters()
        qp.makeDummiesQueries=True
        # Renumber based on frags (incl. dummy atoms)
        aligned_mols = []

        sub_idx = []
        # Align to frags and linker
        qfrag = Chem.AdjustQueryProperties(frags,qp)
        frags_matches = list(mol.GetSubstructMatches(qfrag, uniquify=False))
        qlinker = Chem.AdjustQueryProperties(linker,qp)
        linker_matches = list(mol.GetSubstructMatches(qlinker, uniquify=False))
            
        # Loop over matches
        for frag_match, linker_match in product(frags_matches, linker_matches):
            # Check if match
            f_match = [idx for num, idx in enumerate(frag_match) if frags.GetAtomWithIdx(num).GetAtomicNum() != 0]
            l_match = [idx for num, idx in enumerate(linker_match) if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 and idx not in f_match]
            if len(set(list(f_match)+list(l_match))) == mol.GetNumHeavyAtoms():
            #if len(set(list(frag_match)+list(linker_match))) == mol.GetNumHeavyAtoms():
                break
        # Add frag indices
        sub_idx += frag_match
        # Add linker indices to end
        sub_idx += [idx for num, idx in enumerate(linker_match) if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 and idx not in sub_idx]

        nodes_to_keep = [i for i in range(len(frag_match))]

        aligned_mols.append(Chem.rdmolops.RenumberAtoms(mol, sub_idx))
        aligned_mols.append(frags)
            
        # Renumber dummy atoms to end
        dummy_idx = []
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_idx.append(atom.GetIdx())
        for i, mol in enumerate(aligned_mols):
            sub_idx = list(range(aligned_mols[1].GetNumHeavyAtoms()+2))
            for idx in dummy_idx:
                sub_idx.remove(idx)
                sub_idx.append(idx)
            if i == 0:
                mol_range = list(range(mol.GetNumHeavyAtoms()))
            else:
                mol_range = list(range(mol.GetNumHeavyAtoms()+2))
            idx_to_add = list(set(mol_range).difference(set(sub_idx)))
            sub_idx.extend(idx_to_add)
            aligned_mols[i] = Chem.rdmolops.RenumberAtoms(mol, sub_idx)
            
        # Get exit vectors
        exit_vectors = []
        linker_atom_idx = []
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                if atom.GetIdx() in nodes_to_keep:
                    nodes_to_keep.remove(atom.GetIdx())
                for nei in atom.GetNeighbors():
                    exit_vectors.append(nei.GetIdx())
                linker_atom_idx.append(atom.GetIdx())
                    
        # Get coords
        conf = aligned_mols[0].GetConformer()
        exit_coords = []
        for exit in exit_vectors:
            exit_coords.append(np.array(conf.GetAtomPosition(exit)))
        linker_coords = []
        for linker_atom in linker_atom_idx:
            linker_coords.append(np.array(conf.GetAtomPosition(linker_atom)))
        
        # Get angle
        v1_u = unit_vector(linker_coords[0]-exit_coords[0])
        v2_u = unit_vector(linker_coords[1]-exit_coords[1])
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
                    
        # Get linker length
        linker = Chem.MolFromSmiles(smi_linker)
        linker_length = linker.GetNumHeavyAtoms()

        # Get distance
        distance = np.linalg.norm(exit_coords[0]-exit_coords[1])
                
        # Record results
        return distance, angle
    
    except:
        print(Chem.MolToSmiles(mol), smi_linker, smi_frags)
        return None, None
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



