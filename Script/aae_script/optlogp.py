'''
Descripttion: 
version: 
Author: 成凯阳
Date: 2022-05-07 08:31:38
LastEditors: 成凯阳
LastEditTime: 2022-09-01 03:28:20
'''


#!/usr/bin/env python





# import sys, os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# import torch
# from torch.utils.data import DataLoader
# import pickle
# from rdkit import Chem
# from rdkit import rdBase
# from tqdm import tqdm
# import numpy as np
# from Dataset.get_dataset import get_dataset,get_lookup_tables
# from Dataset.get_Vocab import AAEVocabDatasets,Vocabulary,collate_fnS
# from Model.aae_model import AAE
# import argparse
# from step2_train_aae import pretrain

# from Utils.utils.train_utils import NoamLR,decrease_learning_rate
# from Utils.utils.metric import fraction_valid_smiles
# from torch.nn import CrossEntropyLoss
# from torch import log_, optim
# import pandas as pd

# import os
# from rdkit import rdBase
# from rdkit.Chem import AllChem,MACCSkeys,Descriptors
# from rdkit import DataStructs

# import re

# from Utils.utils.scorefunction import pair_log



# from torch.nn.utils.rnn import pad_sequence
# from collections import Counter
# from Utils.utils.data_utils import Variable
# from rdkit.Chem import MolFromSmiles

# rdBase.DisableLog('rdApp.error')


# def add_optimizer_args(parser):
#     group = parser.add_argument_group("optimizering options")
#     group.add_argument("--restore_from", help="Checkpoint to load", type=bool, default=False)
#     group.add_argument("--hidden", help="Model hidden size", type=int, default="128")
  
   
#     group.add_argument('--latent_size', type=int, default=128,
#                            help='Size of latent vectors')
#     group.add_argument('--embedding_size', type=int, default=32,
#                            help='Embedding size in encoder and decoder')
#     group.add_argument('--decoder_hidden_size', type=int, default=512,
#                            help='Size of hidden state for lstm '
#                                 'layers in decoder')
#     group.add_argument('--encoder_bidirectional', type=bool, default=True,
#                            help='If true to use bidirectional lstm '
#                                 'layers in encoder')
#     group.add_argument('--discriminator_layers', nargs='+', type=int,
#                            default=[640, 256],
#                            help='Numbers of features for linear '
#                                 'layers in discriminator')
#     group.add_argument('--encoder_hidden_size', type=int, default=512,
#                            help='Size of hidden state for '
#                                 'lstm layers in encoder')
#     group.add_argument('--discriminator_steps', type=int, default=1,
#                            help='Discriminator training steps per one'
#                                 'autoencoder training step')
#     group.add_argument('--decoder_num_layers', type=int, default=2,
#                            help='Number of lstm layers in decoder')
  

#     group.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
#     group.add_argument("--beta1", help="Adam beta 1", type=float, default=0.9)
#     group.add_argument("--beta2", help="Adam beta 2", type=float, default=0.998)
#     group.add_argument("--eps", help="Adam epsilon", type=float, default=1e-9)
#     group.add_argument("--weight_decay", help="Adam weight decay", type=float, default=1e-4)
#     group.add_argument("--save_logs", help="path to save logs", type=str, default='/home/chengkaiyang/Main/optdouble/aae/optmetrics.csv')
#     group.add_argument("--gpu", help="use gpu", type= int, default='1')
#     group.add_argument("--max_length", help="max length to sample", type=int, default=100)
#     group.add_argument("--n_batch", help="batch size to sample", type=int, default=1000)
#     group.add_argument("--num_layers", help="Model layers", type=int, default="2")
#     # group.add_argument("--train_p", help="Checkpoint to load", type=str, default='trainfilter.csv')
#     group.add_argument("--save_logss", help="path to save check", type=str, default='/home/chengkaiyang/Main/optdouble/aae/metric.csv')
#     group.add_argument("--save_dir", help="path to save check", type=str, default='/home/chengkaiyang/Main/optdouble/aae/')
#     group.add_argument("--dropout", help="random sample point ", type=float, default="0.")
#     group.add_argument("--restore_agent_from", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/savehuizong/aae/aaemodels.20.pt')
#     group.add_argument("--vocab_path", help="Vocab path to load", type=str, default='/home/chengkaiyang/Main/datanew/data/Voc.txt')
#     group.add_argument("--score_function", help="score function to choose", type=str, default='fingerprint')
#     group.add_argument("--testmol", help="test mol similarity ", type=str, default='CC(C)(C)OC(=O)N1CC[NH2+]CC1C(N)=O')
#     group.add_argument("--cuda", help="use gpu device", type=str, default='cuda:0')
#     group.add_argument("--sim", help="similarity", type=float, default=0.2)
#     group.add_argument("--sthreshold", help="similarity threshold", type=float, default=0.7)
#     group.add_argument("--lthreshold", help="logp threshold", type=float, default=3.5)
   

  
#     return group


# def reback(model,toens,n_batch):
#     smis=[]
   

#     for i in range(n_batch):
#         stri=''
#         for j in range(len(toens[i].cuda().cpu().numpy())):
#             smi=model.vocabulary.reversed_vocab[int(toens[i][j])]
#             stri+=smi
#         stri=stri.split('<pad>')[0]
#         if Chem.MolFromSmiles(stri):
#             smis.append(stri)

#         # smis.append(stri)
       

        

#     return smis


# def train_agent(query_fp,args,
#                 restore_agent_from=None,
#                 scoring_function=None,
#                 scoring_function_kwargs=None,
#                 save_dir=None, learning_rate=0.0005,
#                 batch_size=None, n_steps=70,
#                 num_processes=0, sigma=100,
#                 experience_replay=0):

#     voc = Vocabulary(init_from_file=args.vocab_path)



   
#     Agent = AAE(voc,args)


#     if args.gpu:
#         device=args.cuda
       
#         Agent=Agent.to(device)
#     else:
     
#         Agent=Agent
#         device='cpu'
#         Agent=Agent.to(device)
#     aoptimizer = optim.Adam(
#           list(Agent.encoder.parameters()) +
#                 list(Agent.decoder.parameters()),
#         lr=args.lr,
    
#         weight_decay=args.weight_decay
#     )
#     # aoptimizer=torch.optim.RMSprop(list(Prior.encoder.parameters()) +
#     #             list(Prior.decoder.parameters()), lr=0.05, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
 
#     doptimizer = optim.Adam(
#         Agent.discriminator.parameters(),
#         lr=args.lr,
     
#         weight_decay=args.weight_decay
#     )
  
#     state = torch.load(restore_agent_from,map_location=device)
#     # pretrain_state_dict = state['state_dict']
#     pretrain_state_dict=state['state_dict']

#     Agent.load_state_dict(pretrain_state_dict)
#     df1 = pd.DataFrame(columns = ['smiles', 'best score','avg score'])
#     df1.to_csv(args.save_logss)

#     print("Model initialized, starting training...")

#     for step in range(n_steps):
#         train_ss=[]
#         val_ss=[]

#         # Sample from Agent
#         new=Agent.sample(n_batch=args.n_batch,args=args)
#     # new_pad=pad_sequence(new,batch_first=True)
    
#         smiles=reback(Agent,new,args.n_batch)
     
#         score=pair_log(smiles)
#         s=np.mean(score)
#         if s>=args.lthreshold:
#             print('get threshold')
#             break
#         max_score=score.index(max(score))
#         max_smiles=smiles[max_score]
#         idx=[i for i,a in enumerate(score) if a >s]
#         # idx = heapq.nlargest(100, range(len( score)),  score.__getitem__)
#         smiles=[smiles[id]  for id in idx]
      
       
       
#         dict1 = {'smiles': smiles} 
#         df = pd.DataFrame(dict1)  
 
#         df.to_csv(args.save_logs,mode='w')
#         moldatatr=AAEVocabDatasets(fname=args.save_logs,voc=voc)
     
  
#         train_data = DataLoader(moldatatr, batch_size=10, shuffle=True, drop_last=True,
#                       collate_fn=collate_fnS)
   
#         for i in tqdm(range(1), desc='Processing'):
#             train_loss,_,_=pretrain(args,Prior=Agent,autooptimizer=aoptimizer,disoptimizer=doptimizer,train_dat=train_data,epoc=i)

      
#         lists = [max_smiles, max(score), s]
#         data = pd.DataFrame([lists])
#         data.to_csv(args.save_logss,mode='a',header=False,index=False)#
#         torch.save(Agent.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))
   

  
  
#     # torch.save(Agent.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))

   

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser("preprocess and optimizer")
#     group=add_optimizer_args(parser)

    

#     args = parser.parse_args()



#     train_agent(query_fp=args.testmol,args=args,save_dir=args.save_dir,restore_agent_from=args.restore_agent_from,scoring_function=args.score_function,scoring_function_kwargs={})
'''
Descripttion: 
version: 
Author: 成凯阳
Date: 2022-05-07 08:31:38
LastEditors: 成凯阳
LastEditTime: 2022-08-31 05:47:05
'''


#!/usr/bin/env python



# from sklearn import metrics

from ast import Break
from multiprocessing.spawn import old_main_modules
import sys, os

from Script.aae_script.opt_sim import train_agent_opt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm
import numpy as np
from Dataset.get_dataset import get_dataset,get_lookup_tables
from Dataset.get_Vocab import AAEVocabDatasets,Vocabulary,collate_fnS
from Model.aae_model import AAE
import argparse
from Script.aae_script.opt_similarity import pretrain
# from step5_opt_sim import train_agents
from Utils.utils.train_utils import NoamLR,decrease_learning_rate
from Utils.utils.metric import fraction_valid_smiles
from torch.nn import CrossEntropyLoss
from torch import log_, optim
import pandas as pd
import random
import os
from rdkit import rdBase
from rdkit.Chem import AllChem,MACCSkeys,Descriptors
from rdkit import DataStructs
from rdkit.Chem import Draw
# import re
from rdkit.Chem import QED
from Utils.utils.metric import  logP,SA
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
# from rdkit.Chem import Draw
# from joblib import Parallel,delayed
# from torch.nn.utils.rnn import pad_sequence
# from collections import Counter
# from Utils.utils.data_utils import Variable
# from rdkit.Chem import MolFromSmiles
from rdkit.Chem import PandasTools, QED, Descriptors, rdMolDescriptors
# from Utils.utils.scorefunction import fp_print,similarity
# from utils import Variable, decrease_learning_rate
#/user-data/Main/save/model.26_10.pt
rdBase.DisableLog('rdApp.error')

   

def fp_print(smiles,query_fp):

    score=[]
    for s in smiles:
        t=similarity(s,query_fp)
   
        score.append(t)
     

   
    return score
def similarity(a, b):
    if a is None or b is None: 
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0

    # if sim_type == "binary":
    #     fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    #     fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    # else:
    fp1 = AllChem.GetMorganFingerprint(amol, 2, useChirality=False)
    fp2 = AllChem.GetMorganFingerprint(bmol, 2, useChirality=False)

    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    
    return sim


# def similarity(a, b):
#     if a is None or b is None: 
#         return 0.0
#     amol = Chem.MolFromSmiles(a)
#     bmol = Chem.MolFromSmiles(b)
#     if amol is None or bmol is None:
#         return 0.0



#     fp1 = MACCSkeys.GenMACCSKeys(amol)
#     fp2 = MACCSkeys.GenMACCSKeys(bmol)
#     return DataStructs.FingerprintSimilarity(fp1, fp2)

def add_optimizer_args(parser):
    group = parser.add_argument_group("optimizering options")

    group.add_argument("--hidden", help="Model hidden size", type=int, default="128")
  
   
    group.add_argument('--latent_size', type=int, default=128,
                           help='Size of latent vectors')
    group.add_argument('--embedding_size', type=int, default=32,
                           help='Embedding size in encoder and decoder')
    group.add_argument('--decoder_hidden_size', type=int, default=512,
                           help='Size of hidden state for lstm '
                                'layers in decoder')
    group.add_argument('--encoder_bidirectional', type=bool, default=True,
                           help='If true to use bidirectional lstm '
                                'layers in encoder')
    group.add_argument('--discriminator_layers', nargs='+', type=int,
                           default=[640, 256],
                           help='Numbers of features for linear '
                                'layers in discriminator')
    group.add_argument('--encoder_hidden_size', type=int, default=512,
                           help='Size of hidden state for '
                                'lstm layers in encoder')
    group.add_argument('--discriminator_steps', type=int, default=1,
                           help='Discriminator training steps per one'
                                'autoencoder training step')
    group.add_argument('--decoder_num_layers', type=int, default=2,
                           help='Number of lstm layers in decoder')
  

    group.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    group.add_argument("--learningrate", help="Learning rate", type=float, default=4e-4)
    group.add_argument("--beta1", help="Adam beta 1", type=float, default=0.9)
    group.add_argument("--beta2", help="Adam beta 2", type=float, default=0.998)
    group.add_argument("--eps", help="Adam epsilon", type=float, default=1e-9)
    group.add_argument("--weight_decay", help="Adam weight decay", type=float, default=1e-4)
    group.add_argument("--save_logs", help="path to save logs", type=str, default='/home/chengkaiyang/Main/opt/aae/moptmetrics.csv')
    group.add_argument("--gpu", help="use gpu", type= int, default='1')
    group.add_argument("--max_length", help="max length to sample", type=int, default=100)
    group.add_argument("--n_batch", help="batch size to sample", type=int, default=2000)
    group.add_argument("--num_layers", help="Model layers", type=int, default="2")
    # group.add_argument("--train_p", help="Checkpoint to load", type=str, default='trainfilter.csv')
    group.add_argument("--save_logss", help="path to save check", type=str, default='/home/chengkaiyang/Main/opt/aae/mmetric.csv')
    group.add_argument("--save_dir", help="path to save check", type=str, default='/home/chengkaiyang/Main/opt/aae/')
    group.add_argument("--dropout", help="random sample point ", type=float, default="0.")
    group.add_argument("--restore_agent_from", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/opt/aae/Agent.ckpt')
    group.add_argument("--vocab_path", help="Vocab path to load", type=str, default='/home/chengkaiyang/Main/datanew/data/Voc.txt')
    group.add_argument("--restore_prior_from", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/savehuizong/aae/aaemodels.20.pt')
    group.add_argument("--testmol", help="test mol similarity ", type=str, default='CC1=NN=C(C2=CC(N3CCOCC3)=NC=C21)N[C@@H](C)C4=CC=CC(C#N)=C4C')
    group.add_argument("--cuda", help="use gpu device", type=str, default='cuda:0')
    group.add_argument("--threshold", help="similarity threshold", type=float, default=0.5)
    group.add_argument("--s1", help="molecular substructure", type=str, default='FC(F)(F)C1=CC(C(C)N)=CC(N)=C1')#Nc1cc(C(N)[CH3:1])cc(C(F)(F)F)c1
    group.add_argument("--s2", help="molecular substructure", type=str, default='OC1COCC1')
    group.add_argument("--generate_file", help="generate sample files", type=str, default='/home/chengkaiyang/Main/savehuizong/aae/aaeSample.txt')
#     # file paths
#     group.add_argument("--train_bin", help="Train npz", type=str, default="")
#     group.add_argument("--valid_bin", help="Valid npz", type=str, default="")
  
    return group


def reback(model,toens,n_batch):
    smis=[]
   

    for i in range(n_batch):
        stri=''
        for j in range(len(toens[i].cpu().numpy())):
            smi=model.vocabulary.reversed_vocab[int(toens[i][j])]
            stri+=smi
        stri=stri.split('<pad>')[0]
        if Chem.MolFromSmiles(stri):
            smis.append(stri)

        # smis.append(stri)
       

        

    return smis
# def save_mol_png(mol, filepath, size=(600, 600)):
#     Draw.MolToFile(mol, filepath, size=size)
def panduan(s,voc):
    lie=[]
    for i in s:
        x=(voc.tokenize('0,'+i))
        if set(x).issubset(voc.vocab):
            lie.append(i)
    return lie

def pro(mol):
    return Chem.MolFromSmiles(mol)
def geiding(old):
    if len(old)>25:
        return 5000
    else:
        return 150
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
def train_agent_opts(input_param,query_fp,args,
                restore_agent_from=None,
            
                scoring_function_kwargs=None,
                save_dir=None, learning_rate=0.0005,
                batch_size=None, n_steps=700000,
                num_processes=0, sigma=100,
                experience_replay=0):

    voc = Vocabulary(init_from_file=args.vocab_path)



   
    Agent = AAE(voc,args)


    if args.gpu:
        device=args.cuda
       
        Agent=Agent.to(device)
    else:
     
        Agent=Agent
        device='cpu'
        Agent=Agent.to(device)
    aoptimizer = optim.Adam(
          list(Agent.encoder.parameters()) +
                list(Agent.decoder.parameters()),
        lr=args.learningrate,
    
        weight_decay=args.weight_decay
    )
    # aoptimizer=torch.optim.RMSprop(list(Prior.encoder.parameters()) +
    #             list(Prior.decoder.parameters()), lr=0.05, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
 
    doptimizer = optim.Adam(
        Agent.discriminator.parameters(),
        lr=args.learningrate,
     
        weight_decay=args.weight_decay
    )
  
    state = torch.load(restore_agent_from,map_location=device)
    pretrain_state_dict = state['state_dict']
    # pretrain_state_dict=state

    Agent.load_state_dict(pretrain_state_dict)



  
  

  


  

    # Information for the logger
    # step_score = [[], []]
    # # df1 = pd.DataFrame(columns = ['step',  'smiles','score','prior','agent','valid_smile'])
    df1 = pd.DataFrame(columns = ['smiles', 'best score','avg score'])
    df1.to_csv(args.save_logss)
    raw_score=0
    raw_smi=query_fp
    # df1s = pd.DataFrame(columns = ['smiles'])
    # df1s.to_csv('/home/chengkaiyang/Main/Script/aae_script/4.csv')

    print("Model initialized, starting training...")
    old=[]
    flag=query_fp.count('@') > 0
    print(flag)
    if flag:

        m = Chem.MolFromSmiles(query_fp)
        query_fp=Chem.MolToSmiles(m, isomericSmiles=False)



    cs=0
    df2 = pd.DataFrame(columns = ["smiles","qed","sa","logp","mw","hbd","hba","rob"])
    df2.to_csv('./aae_script/renwuyi.csv',mode='a',header=["smiles","qed","sa","logp","mw","hbd","hba","rob"],index=False)#
    for step in range(n_steps):
        train_ss=[]
        val_ss=[]
        batchs=geiding(old)

        # Sample from Agent
        new=Agent.sample(n_batch=batchs,args=args)
    # new_pad=pad_sequence(new,batch_first=True)
    
        smiles=reback(Agent,new,batchs)
      
        score=fp_print(smiles,query_fp)
        idx=[i for i,a in enumerate(score) if a >=input_param['threhold']]
        smila=[smiles[id]  for id in idx]
        smila=list(set(smila))
        old_main_mol=smila

        for i in old_main_mol:
            # if i not in old:
            old.append(i)
            s_i=similarity(i, query_fp)
            
            if flag:
                mols=Chem.MolFromSmiles(i)

                # dec_isomers = list(EnumerateStereoisomers(mol))
                try:
                    mols=Chem.MolFromSmiles(i)
                    dec_isomers = list(EnumerateStereoisomers(mols))
                    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(
                    mol, isomericSmiles=True)) for mol in dec_isomers]
                    smiles_3d = [Chem.MolToSmiles(mol, isomericSmiles=True)for mol in dec_isomers]
                    mw,logp,hbd,hba,psa,rob,qed,chiral_center=cal_mol_props(i, verbose=False)
                    # lista = [smiles_3d[0],QED.qed(mols),SA(mols),logP(mols)]
                    lista = [smiles_3d[0],qed,SA(mols),logp,mw,hbd,hba,rob]
                    # si_l=[similarity(smiles_3d[0], query_fp)]
                except:
                    mols=Chem.MolFromSmiles(i)
                    mw,logp,hbd,hba,psa,rob,qed,chiral_center=cal_mol_props(i, verbose=False)
                    lista = [i,qed,SA(mols),logp,mw,hbd,hba,rob]


            else:
                mols=Chem.MolFromSmiles(i)
                mw,logp,hbd,hba,psa,rob,qed,chiral_center=cal_mol_props(i, verbose=False)
                lista = [i,qed,SA(mols),logp,mw,hbd,hba,rob]
            datass = pd.DataFrame([lista])
            
            datass.to_csv('./aae_script/renwuyi.csv',mode='a',header=False,index=False)#
            data_di=args.tmp_dir
            s_i = ' {:.2f}'.format(s_i)
            filepath=os.path.join(data_di, '{}.png'.format(cs))
            cs+=1
            mos=pro(i)
            Draw.MolToFile(mos, filepath, size=(400,400),legend=s_i)
            if len(old)>=input_param['n_batchs']:
                # print('Load trained model and generate data done! Time {:.2f} seconds'.format(time.time() - start))
                sys.exit(0)
       
        smiles=[]
        if len(old)>3000:
            old_new=random.sample(old, 3000)
        # elif len(old) <50:
        #     old_new=random.sample(old, int(1/2*len(old)))
        # elif len(old) <25:
        #     old_new=random.sample(old, int(1/3*len(old)))
        # elif len(old) <12:
        #     old_new=random.sample(old, int(1/4*len(old)))
        else:
            old_new=old
            
        
        smiles.extend(old_new)
        if len(old)==0:
            count=10
        else:
            count=1
        # smiles.extend(old)
        for i in range(count):
            # smiles.extend(smila)
            smiles.append(query_fp)
        # smiles=panduan(smiles,voc)
    
           
      
       
       
        dict1 = {'smiles': smiles} 
        df = pd.DataFrame(dict1)  
 
        df.to_csv(args.save_logs,mode='w')
        # if len(smila)>=int(args.n_batch//2):
        #     print('get terminal')
        #     news=Agent.sample(n_batch=args.n_batch,args=args)
        #     smiless=reback(Agent,news,args.n_batch)
        #     f=open(args.generate_file,"w")
 
        #     for line in smiless:

        #         f.write(line+'\n')
        #     break
        moldatatr=AAEVocabDatasets(fname=args.save_logs,voc=voc)
        print(len(smiles))
        if len(smiles)>50:
            bat=int(1/2*len(smiles))
        elif len(smiles)<=2:
            bat=1
        else:
            bat=int(1/3*len(smiles))
            
     
  
        train_data = DataLoader(moldatatr, batch_size=bat, shuffle=True, drop_last=True,
                      collate_fn=collate_fnS)
   
        for i in tqdm(range(1), desc='Processing'):
            train_loss=pretrain(args,Prior=Agent,autooptimizer=aoptimizer,disoptimizer=doptimizer,train_dat=train_data,epoc=i)

      
            # with open(os.path.join(save_dir, "validsampled"), 'a') as f:
            #     f.write("{}\n".format(train_loss))
        # with open(os.path.join(save_dir, "valid"), 'a') as f:

        #     f.write("{} {:5.2f} {:6.2f}\n".format(max_smiles, max(score), s))
        lists = [len(old)]
        data = pd.DataFrame([lists])
        data.to_csv(args.save_logss,mode='a',header=False,index=False)#
        
        # torch.save(Agent.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))
        # if len(old)>=250000:
        #     from Utils.torch_jtnn.chemutils import get_sanitize,decode_stereo
        #     choice_ls=[0,1]
        #     print('Load trained model and generate data done! Time {:.2f} seconds'.format(time.time() - start))
        #     f=open('/home/chengkaiyang/Main/Script/aae_script/1.txt',"w")
        #     for line in old:
        #         random_choice = np.random.choice(choice_ls, 1)[0]
        #         if random_choice ==1:
        #             line=decode_stereo(line)
        #         else:
        #             line=line

                
        #         f.write(line+'\n')
        #     f.close()
 
    
            
            # for ind,i in enumerate(old):

            #    mol=Chem.MolFromSmiles(i)
            #    data_di='/home/chengkaiyang/Main/Script/aae_script/gen'
            #    filepath=os.path.join(data_di, '{}.png'.format(ind))
            #    Draw.MolToFile(mol, filepath, size=(300,300))
            # break

    
   

  
  
    # torch.save(Agent.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser("preprocess and train")
    group=add_optimizer_args(parser)

    
    # parsing.add_train_args(parser)
    # parser.add_argument_group
    args = parser.parse_args()
    import time
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))
    #train_agents(query_fp=args.testmol,args=args,save_dir=args.save_dir,restore_agent_from=args.restore_prior_from,scoring_function_kwargs={})



    train_agent_opts(start=start,query_fp=args.testmol,args=args,save_dir=args.save_dir,restore_agent_from=args.restore_prior_from,scoring_function_kwargs={})
