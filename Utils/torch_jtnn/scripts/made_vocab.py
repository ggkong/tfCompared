from torch_jtnn.mol_tree import MolTree
from multiprocessing import Pool
from argparse import ArgumentParser
from pathlib import Path
import rdkit
import collections 

class MakeVocab():
    """ Make of vocabulary from smiles file.
    

    Args:
        smiles_path (str): 
        output_path (str):

    """
    def __init__(self,smiles_path,output_path):
        self.smiles_path = Path(smiles_path)
        self.output_path = Path(output_path)

    def __call__(self):

        assert self.smiles_path.is_file() is True ,"{} is not file.".format(self.smiles_path.name)

        with self.smiles_path.open('r') as f:
            smiles_list = [one.strip() for one in f.readlines() if one != '\n']
        print("number of smiles: {}".format(len(smiles_list)))
        self.cset = set()

        with Pool() as p:
            r = p.map_async(self.get_vocab,smiles_list,callback=self.callback)
            while not r.ready():
                print("Now Processing")
                r.wait(timeout=60)

            if r.successful() == False:
                print("Processing Error")
                print(r.get())
        print("Number of Vocab: {}".format(len(self.cset)))
        with self.output_path.open("w") as f:
            for key,count in self.count.items():
                f.write(key+','+str(count)+'\n')

    def get_vocab(self,smiles):
        cset = set()
        try:
            mol = MolTree(smiles)
        except Exception as e:
            pass
        else:
            smiles_list = [c.smiles for c in mol.nodes]
            cset = collections.Counter(smiles_list)
        return cset

    def callback(self,one_set):
        result = dict()
        for one in one_set:
            for key,value in one.items():
                if key not in result:
                    result[key] = 0
                result[key] += value
        
        self.cset |= set(result.keys())
        self.count = result


def make_vocab(smiles_path,output_path):
    """ make vocabrary scripts.
    
    Args:
        smiles_path (str): 
        output_path (str):
    """
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    MakeVocab(smiles_path,output_path)()

def main():
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    parser = ArgumentParser()
    parser.add_argument("smiles_path")
    parser.add_argument("output_path")

    args = parser.parse_args()

    proc = MakeVocab(**args.__dict__)
    proc()


if __name__=="__main__":
    main()
