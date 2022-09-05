import numpy as np
from rdkit import Chem
from typing import List
import re
import torch
data_name = 'ours'
from rdkit.Chem import AllChem
# atom_decoder_m = {0: 6, 1: 7, 2: 8, 3: 9}
atom_decoder_m = {0: 6, 1: 7, 2: 8, 3: 9, 4:15, 5:16, 6:17, 7:35, 8:53}
bond_decoder_m = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
# ATOM_VALENCY = {6:4, 7:3, 8:2, 9:1, 15:3, 16:2, 17:1, 35:1, 53:1}

# OURDATA
# atom_decoder_m = {0: 5, 1: 6, 2: 7, 3: 8, 4:9, 5:14, 6:15, 7:16, 8:17, 9:35}
# bond_decoder_m = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.AROMATIC, 3: Chem.rdchem.BondType.DOUBLE}
ATOM_VALENCY = {5:3, 6:4, 7:3, 8:2, 9:1, 14:4, 15:3, 16:2, 17:1, 35:1, 53:1}

def flatten_graph_data(adj, x):
    return torch.cat((adj.reshape([adj.shape[0], -1]), x.reshape([x.shape[0], -1])), dim=1)


def split_channel(x):
    n = x.shape[1] // 2
    return x[:, :n], x[:, n:]


def get_graph_data(x, num_nodes, num_relations, num_features):
    """
    Converts a vector of shape [b, num_nodes, m] to Adjacency matrix
    of shape [b, num_relations, num_nodes, num_nodes]
    and a feature matrix of shape [b, num_nodes, num_features].
    :param x:
    :param num_nodes:
    :param num_relations:
    :param num_features:
    :return:
    """
    adj = x[:, :num_nodes*num_nodes*num_relations].reshape([-1, num_relations, num_nodes, num_nodes])
    feat_mat = x[:, num_nodes*num_nodes*num_relations:].reshape([-1, num_nodes, num_features])
    return adj, feat_mat


def Tensor2Mol(A, x):
    mol = Chem.RWMol()
    # x[x < 0] = 0.
    # A[A < 0] = -1
    # atoms_exist = np.sum(x, 1) != 0
    atoms = np.argmax(x, 1)
    atoms_exist = atoms != 4
    atoms = atoms[atoms_exist]
    atoms = atom_decoder_m[atoms]
#     atoms += 6
    adj = np.argmax(A, 0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    # print('num atoms: {}'.format(sum(atoms>0)))

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atom)))

    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])

    return mol


def construct_mol(x, A, atomic_num_list):
    """

    :param x:  (9,5)
    :param A:  (4,9,9)
    :param atomic_num_list: [6,7,8,9,0]
    :return:
    """
    mol = Chem.RWMol()
    # x (ch, num_node)
    atoms = np.argmax(x, axis=1)
    # last a
    atoms_exist = atoms != len(atomic_num_list) - 1
    atoms = atoms[atoms_exist]
    # print('num atoms: {}'.format(sum(atoms>0)))

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    # A (edge_type, num_node, num_node)
    adj = np.argmax(A, axis=0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])
            # add formal charge to atom: e.g. [O+], [N+] [S+]
            # not support [O-], [N-] [S-]  [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            # we can process error info like
            # Explicit valence for atom # 0 S, 41, is greater than permitted
            # we cant process below
            # Explicit valence for aromatic atom # 1 not equal to any accepted valence
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if data_name != 'ours':
                    if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                        mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def construct_mol_with_validation(x, A, atomic_num_list):
    """

    :param x:  (9,5)
    :param A:  (4,9,9)
    :param atomic_num_list: [6,7,8,9,0]
    :return:
    """
    mol = Chem.RWMol()
    # x (ch, num_node)
    atoms = np.argmax(x, axis=1)
    # last a
    atoms_exist = atoms != len(atomic_num_list) - 1
    atoms = atoms[atoms_exist]
    # print('num atoms: {}'.format(sum(atoms>0)))

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    # A (edge_type, num_node, num_node)
    adj = np.argmax(A, axis=0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])
            t = adj[start, end]
            while not valid_mol_can_with_seg(mol):
                mol.RemoveBond(int(start), int(end))
                t = t-1
                if t >= 1:
                    mol.AddBond(int(start), int(end), bond_decoder_m[t])

    return mol


def valid_mol(x):
    s = Chem.MolFromSmiles(Chem.MolToSmiles(x, isomericSmiles=True)) if x is not None else None
    if s is not None and '.' not in Chem.MolToSmiles(s, isomericSmiles=True):
        return s
    return None


def valid_mol_can_with_seg(x, largest_connected_comp=True):
    # mol = None
    if x is None:
        return None
    sm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = Chem.MolFromSmiles(sm)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    return mol


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(x):
    if x==None:
        x='C'
        x=Chem.MolFromSmiles(x)
    xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = x
    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len (atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                if queue[0][1] == 12:
                    t = queue[0][1] - 11
                else:
                    t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder_m[t])
                # if '.' in Chem.MolToSmiles(mol, isomericSmiles=True):
                #     print(tt)
                #     print(Chem.MolToSmiles(mol, isomericSmiles=True))

    return mol





def check_tensor(x):
    return valid_mol(Tensor2Mol(*x))


def adj_to_smiles(adj, x, atomic_num_list):
    # adj = _to_numpy_array(adj, gpu)
    # x = _to_numpy_array(x, gpu)
    valid = [Chem.MolToSmiles(construct_mol(x_elem, adj_elem, atomic_num_list), isomericSmiles=True)
             for x_elem, adj_elem in zip(x, adj)]
    return valid





def check_novelty(gen_smiles, train_smiles, n_generated_mols): # gen: say 788, train: 120803
    if len(gen_smiles) == 0:
        novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]  # [1]*45
        novel = len(gen_smiles) - sum(duplicates)  # 788-45=743
        novel_ratio = novel*100./len(gen_smiles)  # 743*100/788=94.289
        abs_novel_ratio = novel*100./n_generated_mols
    print("novelty: {:.3f}%, abs novelty: {:.3f}%".format(novel_ratio, abs_novel_ratio))
    return novel_ratio, abs_novel_ratio
def get_ECFP4(mol):
    ''' Return rdkit ECFP4 fingerprint object for mol
    Parameters: 
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object  
    Returns: 
    rdkit ECFP4 fingerprint object for mol
    '''
    return AllChem.GetMorganFingerprint(mol, 2)