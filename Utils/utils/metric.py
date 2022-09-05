
import imp
import sys, os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from collections import Counter
from functools import partial
import numpy as np
import pandas as pd
# from pyrsistent import v
import scipy.sparse
from scipy.stats import wasserstein_distance
import torch
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import AllChem,MolFromSmiles
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors
# from moses.metrics.SA_Score import sascorer
# from moses.metrics.NP_Score import npscorer
from Utils.utils.sascore import sascore
from torch.autograd import Variable
from rdkit import DataStructs
from scipy.spatial.distance import cosine as cos_distance
import rdkit
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix
# from pysmiles import read_smiles

import pickle
import gzip
import networkx as nx


from rdkit.Chem import Crippen



import math
import numpy as np

# NP_model = pickle.load(gzip.open('data/NP_score.pkl.gz'))
file_path=os.path.abspath(os.path.dirname(__file__))
os_s=os.path.join(file_path,'sascore/fpscores.pkl.gz')
SA_model = {i[j]: float(i[0]) for i in pickle.load(gzip.open(os_s)) for j in range(1, len(i))}
def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def get_reference_data( data):
    ref_smiles = remove_invalid(data, canonize=True, n_jobs=2)
    ref_mols = mapper(2)(get_mol, ref_smiles)
    return ref_smiles, ref_mols
def logP(mol):
    """
    Computes RDKit's logP
    """
    return Chem.Crippen.MolLogP(mol)
def mol_passes_filters(mol,
                       allowed=None,
                       isomericSmiles=False):
    """
    Checks if mol
    * passes MCF and PAINS filters,
    * has only allowed atoms
    * is not charged
    """
    allowed = allowed or {'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'}
    mol = get_mol(mol)
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(
            len(x) >= 8 for x in ring_info.AtomRings()
    ):
        return False
    h_mol = Chem.AddHs(mol)
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
        return False
   
    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    if smiles is None or len(smiles) == 0:
        return False
    if Chem.MolFromSmiles(smiles) is None:
        return False
    return True
def fraction_passes_filters(gen, n_jobs=1):
    """
    Computes the fraction of molecules that pass filters:
    * MCF
    * PAINS
    * Only allowed atoms ('C','N','S','O','F','Cl','Br','H')
    * No charges
    """
    passes = mapper(n_jobs)(mol_passes_filters, gen)
    return np.mean(passes)
def remove_invalid(gen, canonize=True, n_jobs=1):
    """
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if
            x is not None]
def SA(mol):
    """
    Computes RDKit's Synthetic Accessibility score
    """
    return sascore.calculateScore(mol)
def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map
def fraction_valid_smiles(smiles):
    """Takes a list of SMILES and returns fraction valid."""
    i = 0
    ko=[]
    mo=[]
    for smile in smiles:
        if Chem.MolFromSmiles(smile):
            i += 1
            ko.append(smile)
            mo.append(Chem.MolFromSmiles(smile))
    return i / len(smiles),ko ,mo




def QED(mol):
    """
    Computes RDKit's QED score
    """
    return qed(mol)


def weight(mol):
    """
    Computes molecular weight for given molecule.
    Returns float,
    """
    return Descriptors.MolWt(mol)


def get_n_rings(mol):
    """
    Computes the number of rings in a molecule
    """
    return mol.GetRingInfo().NumRings()


def fragmenter(mol):
    """
    fragment mol using BRICS and return smiles list
    """
    fgs = AllChem.FragmentOnBRICSBonds(get_mol(mol))
    fgs_smi = Chem.MolToSmiles(fgs).split(".")
    return fgs_smi


def compute_fragments(mol_list, rows,n_jobs=1):
    """
    fragment list of mols using BRICS and return smiles list
    """
    fragments = Counter()
    for mol_frag in mapper(n_jobs)(fragmenter, mol_list):
        fragments.update(mol_frag)
    f=Counter()
    for mol_frag in mapper(n_jobs)(fragmenter, rows):
        f.update(mol_frag)

    if len(fragments) == 0 or len(f) == 0:

        return np.nan
    keys = np.unique(list(fragments.keys()) + list(f.keys()))
    ref_vec = np.array([fragments.get(k, 0) for k in keys])
    gen_vec = np.array([f.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)
def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:

     sim = <r, g> / ||r|| / ||g||
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)

def compute_scaffolds(mol_list, n_jobs=1, min_rings=2):
    """
    Extracts a scafold from a molecule in a form of a canonic SMILES
    """
    scaffolds = Counter()
    map_ = mapper(n_jobs)
    scaffolds = Counter(
        map_(partial(compute_scaffold, min_rings=min_rings), mol_list))
    if None in scaffolds:
        scaffolds.pop(None)
    return scaffolds


def compute_scaffold(mol, min_rings=2):
    mol = get_mol(mol)
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '' or n_rings < min_rings:
        return None
    return scaffold_smiles

# def tonimato(smile,k):
#     query_structure = "COc1ccccc1OC(=O)Oc1ccccc1OC"
#     query_fp = AllChem.GetMorganFingerprint(MolFromSmiles(query_structure), 2, useCounts=True, useFeatures=True)
#     mol = Chem.MolFromSmiles(smile)
#     if mol:
#         fp = AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)
#         score = DataStructs.TanimotoSimilarity(query_fp, fp)
#         score = min(score, k) / k
#         return float(score)
#     return 0.0


def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=5000, agg='max',
                         device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return agg_tanimoto,np.mean(agg_tanimoto)


def fingerprint(smiles_or_mol, fp_type='maccs', dtype=None, morgan__r=2,
                morgan__n=1024, *args, **kwargs):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits

    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol, *args, **kwargs)
    if molecule is None:
        return None
    if fp_type == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == 'morgan':
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n),
                                 dtype='uint8')
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint
def fingerprints(smiles_mols_array, n_jobs=1, already_unique=False, *args,
                 **kwargs):
    '''
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parralel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    '''
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array,
                                                 return_inverse=True)

    fps = mapper(n_jobs)(
        partial(fingerprint, *args, **kwargs), smiles_mols_array
    )

    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
           for fp in fps]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique:
        return fps[inv_index]
    return fps
class Metric:
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen)

    def precalc(self, moleclues):
        raise NotImplementedError

    def metric(self, pref, pgen):
        raise NotImplementedError
class WassersteinMetric(Metric):
    def __init__(self, func=None, **kwargs):
        self.func = func
        super().__init__(**kwargs)

    def precalc(self, mols):
        if self.func is not None:
            values = mapper(self.n_jobs)(self.func, mols)
        else:
            values = mols
        return {'values': values}

    def metric(self, pref, pgen):
        return wasserstein_distance(
            pref['values'], pgen['values']
        )
def tonimatos(smile_list):
    liebi=[]
    for smiles in smile_list:
        # a=Chem.MolFromSmiles(smiles)
        try:
            a=Chem.MolFromSmiles(smiles)
            x=MACCSkeys.GenMACCSKeys(a)
            liebi.append(x)
        except:
            pass
        
      



    
   
    return liebi
def penalized_logp(s):
    if s is None: return -100.0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return -100.0

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA=-sascore.calculateScore(mol)
    
    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    
    return normalized_log_p + normalized_SA + normalized_cycle
def avesimilarity(gen_p,row_p):
   

    s=[]
    for i in gen_p:
        score=0
        # s=[]

        for j in row_p:

            scores=DataStructs.FingerprintSimilarity(i,j)
            score=max(scores,score)
        # score/=len(row_p)
        s.append(score)

    total=np.mean(s)
    return total
def internal_diversity(gen, n_jobs=1, device='cpu', fp_type='morgan',
                       gen_fps=None, p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps,
                                     agg='mean', device=device, p=p)).mean()

def NLLLosss(inputs, targets):
    target_expanded = torch.zeros(inputs.size()).cuda()
    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss
def novelty(gen, train, n_jobs=1):
    gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return len(gen_smiles_set - train_set) / len(gen_smiles_set)
