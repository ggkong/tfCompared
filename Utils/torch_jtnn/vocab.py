import rdkit
import rdkit.Chem as Chem
import copy
import numpy as np

def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]

class Vocab(object):
    """ The Vocabraries class.
    
    Args:
        smiles_list (list of str): [description]

    """
    benzynes = ['C1=CC=CC=C1', 'C1=CC=NC=C1', 'C1=CC=NN=C1', 'C1=CN=CC=N1', 'C1=CN=CN=C1', 'C1=CN=NC=N1', 'C1=CN=NN=C1', 'C1=NC=NC=N1', 'C1=NN=CN=N1']
    penzynes = ['C1=C[NH]C=C1', 'C1=C[NH]C=N1', 'C1=C[NH]N=C1', 'C1=C[NH]N=N1', 'C1=COC=C1', 'C1=COC=N1', 'C1=CON=C1', 'C1=CSC=C1', 'C1=CSC=N1', 'C1=CSN=C1', 'C1=CSN=N1', 'C1=NN=C[NH]1', 'C1=NN=CO1', 'C1=NN=CS1', 'C1=N[NH]C=N1', 'C1=N[NH]N=C1', 'C1=N[NH]N=N1', 'C1=NN=N[NH]1', 'C1=NN=NS1', 'C1=NOC=N1', 'C1=NON=C1', 'C1=NSC=N1', 'C1=NSN=C1']

    def __init__(self, smiles_list,weightes = None):
        self.vocab = smiles_list
        self.weightes = weightes if weightes is not None else np.ones(len(smiles_list))

        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]

        Vocab.benzynes = [s for s in smiles_list if s.count('=') >= 2 and Chem.MolFromSmiles(s).GetNumAtoms() == 6] + ['C1=CCNCC1']
        Vocab.penzynes = [s for s in smiles_list if s.count('=') >= 2 and Chem.MolFromSmiles(s).GetNumAtoms() == 5] + ['C1=NCCN1','C1=NNCC1']
        
    def get_index(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]
    def get_weight(self,idx):
        return self.weightes[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)

class ModfVocab(object):
    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        if type(smiles_list[0]) is str:
            self.max_len = max([Chem.MolFromSmiles(smiles).GetNumAtoms() for smiles in smiles_list])
        
    def __getitem__(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def size(self):
        return len(self.vocab)

        

COMMON_ATOMS = [('B', 0), ('B', -1), ('Br', 0), ('Br', -1), ('Br', 2), ('C', 0), ('C', 1), ('C', -1), ('Cl', 0), ('Cl', 1), ('Cl', -1), ('Cl', 2), ('Cl', 3), ('F', 0), ('F', 1), ('F', -1), ('I', -1), ('I', 0), ('I', 1), ('I', 2), ('I', 3), ('N', 0), ('N', 1), ('N', -1), ('O', 0), ('O', 1), ('O', -1), ('P', 0), ('P', 1), ('P', -1), ('S', 0), ('S', 1), ('S', -1), ('Se', 0), ('Se', 1), ('Se', -1), ('Si', 0), ('Si', -1)]
common_atom_vocab = ModfVocab(COMMON_ATOMS)