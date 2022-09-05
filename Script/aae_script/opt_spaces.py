'''
Descripttion: 
version: 
Author: 成凯阳
Date: 2022-05-07 08:31:38
LastEditors: 成凯阳
LastEditTime: 2022-08-31 12:58:46
'''


#!/usr/bin/env python



# from sklearn import metrics

from multiprocessing.spawn import old_main_modules
import sys, os

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
from rdkit.Chem import Draw
from Script.aae_script.opt_similarity import pretrain
from Script.aae_script.opt_similarity import train_agents
# from Utils.utils.train_utils import NoamLR,decrease_learning_rate
# from Utils.utils.metric import fraction_valid_smiles
from rdkit.Chem import PandasTools, QED, Descriptors, rdMolDescriptors
from torch import  optim
import pandas as pd
from rdkit.Chem import QED
from Utils.utils.metric import  logP,SA
from Utils.torch_jtnn.chemutils import get_sanitize,decode_stereo
from Utils.utils.metric import average_agg_tanimoto,fingerprints
from rdkit import rdBase
from rdkit.Chem import AllChem,MACCSkeys,Descriptors
from rdkit import DataStructs
from rdkit.DataStructs import TanimotoSimilarity
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



    fp1 = MACCSkeys.GenMACCSKeys(amol)
    fp2 = MACCSkeys.GenMACCSKeys(bmol)
    return DataStructs.FingerprintSimilarity(fp1, fp2)

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
  

    # group.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    group.add_argument("--learningrate", help="Learning rate", type=float, default=3e-4)
    group.add_argument("--beta1", help="Adam beta 1", type=float, default=0.9)
    group.add_argument("--beta2", help="Adam beta 2", type=float, default=0.998)
    group.add_argument("--eps", help="Adam epsilon", type=float, default=1e-9)
    group.add_argument("--weight_decay", help="Adam weight decay", type=float, default=1e-4)
    group.add_argument("--save_logs", help="path to save logs", type=str, default='/home/chengkaiyang/new/models/optmetrics.csv')
    group.add_argument("--gpu", help="use gpu", type= int, default='1')
    group.add_argument("--max_length", help="max length to sample", type=int, default=100)
    #group.add_argument("--n_batch", help="batch size to sample", type=int, default=20)
    group.add_argument("--n_batchs", help="final batch size to sample", type=int, default=200)
    group.add_argument("--num_layers", help="Model layers", type=int, default="2")
    # group.add_argument("--train_p", help="Checkpoint to load", type=str, default='trainfilter.csv')
    group.add_argument("--save_logss", help="path to save metric", type=str, default='/home/chengkaiyang/new/models/metric.csv')
    #group.add_argument("--save_dir", help="path to save check", type=str, default='/home/chengkaiyang/Main/opt/aae/')
    group.add_argument("--dropout", help="random sample point ", type=float, default="0.")
    group.add_argument("--restore_agent_from", help="Checkpoint to load", type=str, default='/home/chengkaiyang/new/models/aaemodels.20.pt')
    group.add_argument("--vocab_path", help="Vocab path to load", type=str, default='/home/chengkaiyang/new/Script/voc.txt')
    #group.add_argument("--restore_prior_from", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/savehuizong/aae/aaemodels.20.pt')
    #group.add_argument("--testmol", help="test mol similarity ", type=str, default='FC(F)(F)C1=CC(C(C)NC2=NC(C)=NC3=C2C=C(OC4COCC4)C(OC)=C3)=CC(N)=C1')
    group.add_argument("--cuda", help="use gpu device", type=str, default='cpu')
    group.add_argument("--tmp_dir", help="tmp dir to save ", type=str, default='/home/chengkaiyang/Main/Script/aae_script/gen')
    group.add_argument("--threshold", help="similarity threshold", type=float, default=0.7)
    group.add_argument("--testfile", help="test moleculars ", type=str, default='Script/1.txt')
    group.add_argument("--savefile", help="save generate moleculars ", type=str, default='Script/2.txt')
    group.add_argument("--result_name", help="logs to save ", type=str, default='Script/aae_script/2.csv')
#     # file paths
#     group.add_argument("--train_bin", help="Train npz", type=str, default="")
#     group.add_argument("--valid_bin", help="Valid npz", type=str, default="")
  
    return group


def reback(model,toens,n_batch):
    smis=[]
    choice_ls=[0,1]

    for i in range(n_batch):
        stri=''
        for j in range(len(toens[i].cpu().numpy())):
            smi=model.vocabulary.reversed_vocab[int(toens[i][j])]
            stri+=smi
        stri=stri.split('<pad>')[0]
        if Chem.MolFromSmiles(stri):
            random_choice = np.random.choice(choice_ls, 1)[0]
            if random_choice==1:
                smis.append(stri)
                # stri=decode_stereo(stri)
                # smis.append(stri[0])
            else:
                smis.append(stri)
            # smis.append(stri)

        # smis.append(stri)
       

        

    return smis
def pro(mol):
    return Chem.MolFromSmiles(mol)
def diversity(mols):
    fp_list = []
    mols=[pro(mol) for mol in mols]
    for molecule in mols:

        fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
        fp_list.append(fp)
    diversity = []
    for i in range(len(fp_list)):
        for j in range(i+1, len(fp_list)):
            current_diverity  = 1 - float(TanimotoSimilarity(fp_list[i], fp_list[j]))
            diversity.append(current_diverity)
    return np.mean(diversity)

def return_csv(generated_all_smiles,path,old):
    import pandas as pd
    # df1 = pd.DataFrame(columns = ['diversity',  'unique','nolvety'],index=0)
    # df1.to_csv(path)
    val=0
    zong=len(generated_all_smiles)
    valid_mol=[Chem.MolFromSmiles(i) for i in generated_all_smiles]
    val_rate=len(valid_mol)/zong
    gen_smiles_set = set(generated_all_smiles) - {None}
    train_set = set(old)
    nolvelty=len(gen_smiles_set - train_set) / len(gen_smiles_set)
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
    list = [div,unique_ratio,nolvelty ]
    d = {'diversity':[div],  'unique':[unique_ratio],'nolvety':[nolvelty]}
    data = pd.DataFrame(data=d, columns = ['diversity',  'unique','nolvety'],dtype=float)


  
    data.to_csv(path,header=True,index=False)#
    return valid_mol
def save_pngs(args,x_smi,score):
    df2 = pd.DataFrame(columns = ["smiles","qed","sa","logp","mw","hbd","hba","rob"])
    df2.to_csv('./aae_script/renwuyi.csv',mode='a',header=["smiles","qed","sa","logp","mw","hbd","hba","rob"],index=False)#
    for ind,(i,s) in enumerate(zip(x_smi,score)):

        mol=Chem.MolFromSmiles(i)
        mw = np.round(Descriptors.MolWt(mol),1)
        logp = np.round(Descriptors.MolLogP(mol),2)
        hbd = rdMolDescriptors.CalcNumLipinskiHBD(mol)
        hba = rdMolDescriptors.CalcNumLipinskiHBA(mol)
        rob= rdMolDescriptors.CalcNumRotatableBonds(mol)
     
        qed= np.round(QED.qed(mol),2)
        lista = [i,qed,SA(mol),logp,mw,hbd,hba,rob]
        # lista = [i]
        datass = pd.DataFrame([lista])
        datass.to_csv('./aae_script/renwuyi.csv',mode='a',header=False,index=False)#

        data_di=args.tmp_dir
        s = ' {:.2f}'.format(s)
        filepath=os.path.join(data_di, '{}.png'.format(ind))
        Draw.MolToFile(mol, filepath, size=(400,400),legend=s)
def train_agent_aae(input_param,args,
                restore_agent_from=None,
            
                scoring_function_kwargs=None,
                learning_rate=0.0005,
                batch_size=None, n_steps=7000,
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
    # pretrain_state_dict = state['state_dict']
    pretrain_state_dict=state['state_dict']

    Agent.load_state_dict(pretrain_state_dict)



  
  

  


  

 

    print("Model initialized, starting training...")
   
    with open(input_param['testfile'], "r") as f:
      old = f.readlines()
    old=[od[:-1] for od in old ]
    print(old)
    old=get_sanitize(old)

    locate=old.copy()
    smiless=[]
   
    for step in range(n_steps):
        train_ss=[]
        val_ss=[]

        # Sample from Agent
        new=Agent.sample(n_batch=5000,args=args)
    # new_pad=pad_sequence(new,batch_first=True)
    
        smiles=reback(Agent,new,5000)

        # smila=list(set(smiles))
        # scores=diversity(smiles)
        score,scores=average_agg_tanimoto(fingerprints(old,n_jobs=1), fingerprints(smiles,n_jobs=1),device=device)
        idx=list(np.where(score>=input_param['threhold'])[0])
        temp_smi=[smiles[i] for i in idx if smiles[i] !='']
        smiless.extend(temp_smi)
        print(len(smiless))
        if len(smiless)>=input_param['n_batchs']:
            smiless=smiless[:input_param['n_batchs']]
        #     new=Agent.sample(n_batch=input_param['n_batchs'],args=args)
        #     smiless=reback(Agent,new,input_param['n_batchs'])
            a=return_csv(smiless,path=args.result_name,old=old)
            score,scores=average_agg_tanimoto(fingerprints(old,n_jobs=4), fingerprints(smiless,n_jobs=4),device=device)
            save_pngs(args,smiless,score)

            # f=open(args.savefile,'w')
            # for line in smiless:
            #     f.write(line+'\n')
            break
            
        # if scores<0.80:
        #     print('get')
        #     f=open(args.savefile,'w')
        #     for line in smiles[:20]:
        #         f.write(line+'\n')
        #     break
        # old_main_mol=smila
        # for i in old_main_mol:
        #     if i not in old:
        #         old.append(i)
        # if step<=10:
        smiles=[]
        smiles.extend(locate)


            
          
        # else:
        #     for i in smila:
        #         if i not in old:
                    
        #             old.append(i)
        #     smiles=[]
        #     smiles.extend(old)


       

   
           
      
       
       
        dict1 = {'smiles': smiles} 
        df = pd.DataFrame(dict1)  
 
        df.to_csv(args.save_logs,mode='w')
        # if len(smila)>=int(args.n_batch//4):
        #     print('get terminal')
        #     break
        moldatatr=AAEVocabDatasets(fname=args.save_logs,voc=voc)
     
  
        train_data = DataLoader(moldatatr, batch_size=5, shuffle=True, drop_last=True,
                      collate_fn=collate_fnS)
   
        for i in tqdm(range(1), desc='Processing'):
            train_loss=pretrain(args,Prior=Agent,autooptimizer=aoptimizer,disoptimizer=doptimizer,train_dat=train_data,epoc=i)

      
            # with open(os.path.join(save_dir, "validsampled"), 'a') as f:
            #     f.write("{}\n".format(train_loss))
        # with open(os.path.join(save_dir, "valid"), 'a') as f:

        #     f.write("{} {:5.2f} {:6.2f}\n".format(max_smiles, max(score), s))
        lists = [scores]
        data = pd.DataFrame([lists])
        data.to_csv(args.save_logss,mode='a',header=False,index=False)#
        # torch.save(Agent.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))
    
   

  
  
    # torch.save(Agent.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))

   

if __name__ == "__main__":
    # parser = argparse.ArgumentParser("preprocess and train")
    # group=add_optimizer_args(parser)

    

    # args = parser.parse_args()
    # args=vars(args)
    import json
    file_path=os.path.abspath(os.path.dirname(__file__))
    # # b = json.dumps(args)
    # # f2 = open('/home/chengkaiyang/Main/Script/aae_script/param.json', 'w')
    # # f2.write(b)
    # # f2.close()
    with open(os.path.join(file_path,'param.json'), 'r') as f:

        args= json.load(f)
    with open(os.path.join(file_path,'params.json'), 'r') as f:

        input_param= json.load(f)


    #train_agents(query_fp=args.testmol,args=args,save_dir=args.save_dir,restore_agent_from=args.restore_prior_from,scoring_function_kwargs={})

    args = argparse.Namespace(**args)
    

    train_agent_aae(input_param=input_param,args=args,restore_agent_from=args.restore_agent_from,scoring_function_kwargs={})
