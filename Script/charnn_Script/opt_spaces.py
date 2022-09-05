from sklearn import metrics

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm
import numpy as np
# from Dataset.get_dataset import get_dataset,get_lookup_tables
from Dataset.get_Vocab import VocabDatasets,Vocabulary,collate_fn
from Model.model import CharRNN
import argparse
import json
from Utils.utils.train_utils import NoamLR,decrease_learning_rate
from Utils.utils.metric import fraction_valid_smiles
from torch.nn import CrossEntropyLoss
from torch import optim
import pandas as pd
from Utils.torch_jtnn.chemutils import get_sanitize,decode_stereo
import os
from rdkit.Chem import Draw
from Utils.utils.metric import average_agg_tanimoto,fingerprints
from Utils.utils.data_utils import fp_print,similarity
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from Utils.utils.data_utils import Variable
from rdkit.Chem import MolFromSmiles
from rdkit import rdBase
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem,MACCSkeys,Descriptors
from rdkit import DataStructs
rdBase.DisableLog('rdApp.error')

#CC(NC(=O)OC(C)(C)C)c1nc(CO)nn1Cc1ccccc1



def add_optimizer_args(parser):
    group = parser.add_argument_group("optimizering options")
 
    group.add_argument("--hidden", help="Model hidden size", type=int, default="256")
  
    # group.add_argument("--test_size", help="sample size", type=int, default="1000")
    group.add_argument("--num_layers", help="Model num layers", type=int, default="3")
    group.add_argument("--dropout", help="random sample point ", type=float, default="0.2")
  

    group.add_argument("--lr", help="Learning rate", type=float, default=2e-4)
    group.add_argument("--beta1", help="Adam beta 1", type=float, default=0.9)
    group.add_argument("--beta2", help="Adam beta 2", type=float, default=0.998)
    group.add_argument("--eps", help="Adam epsilon", type=float, default=1e-9)
    group.add_argument("--weight_decay", help="Adam weight decay", type=float, default=1e-2)
    group.add_argument("--save_logs", help="path to save logs", type=str, default='/home/chengkaiyang/Main/opt/charnn/optmetrics.csv')
    group.add_argument("--gpu", help="Adam weight decay", type=str, default='True')
    group.add_argument("--max_length", help="max length to sample", type=int, default=100)
    group.add_argument("--n_batch", help="batch size to sample", type=int, default=20)
    group.add_argument("--n_batchs", help="batch size to sample", type=int, default=200)
    # group.add_argument("--train_p", help="Checkpoint to load", type=str, default='trainfilter.csv')
    # group.add_argument("--valid_p", help="Checkpoint to load", type=str, default='testfilter.csv')
    #group.add_argument("--save_dir", help="path to save check", type=str, default='/home/chengkaiyang/Main/opt/charnn')
    group.add_argument("--tmp_dir", help="tmp dir to save ", type=str, default='/home/chengkaiyang/Main/Script/charnn_Script/gen')
    group.add_argument("--restore_agent_from", help="raw Checkpoint to load", type=str, default='/home/chengkaiyang/Main/s/model.2.pt')
    group.add_argument("--restore_prior_from", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/s/model.199.pt')
    group.add_argument("--vocab_path", help="Vocab path to load", type=str, default='/home/chengkaiyang/Main/datanew/data/Voc.txt')
    group.add_argument("--threshold", help="similarity threshold", type=float, default=0.7)
    group.add_argument("--testfile", help="test moleculars ", type=str, default='/home/chengkaiyang/Main/data/1.txt')
    group.add_argument("--savefile", help="save generate moleculars ", type=str, default='/home/chengkaiyang/Main/data/2.txt')
    group.add_argument("--cuda", help="use gpu device", type=str, default='cpu')
    group.add_argument("--result_name", help="logs to save ", type=str, default='/home/chengkaiyang/Main/Script/charnn_Script/2.csv')
  
    return group
def reback(model,toens,n_batch):
    smis=[]
    
    choice_ls=[0,1]
    for i in range(n_batch):
        stri=''
        for j in range(len(toens[i].cpu().numpy())):
            smi=model.vocabulary.reversed_vocab[int(toens[i][j])]
            stri+=smi
        if Chem.MolFromSmiles(stri):
            random_choice = np.random.choice(choice_ls, 1)[0]
            if random_choice==1:
                smis.append(stri)
                # stri=decode_stereo(stri)
                # smis.append(stri[0])
            else:
                smis.append(stri)
        
    
       

        

    return smis  
def save_pngs(args,x_smi,score):
    for ind,(i,s) in enumerate(zip(x_smi,score)):

        mol=Chem.MolFromSmiles(i)
        data_di=args.tmp_dir
        s = ' {:.2f}'.format(s)
        filepath=os.path.join(data_di, '{}.png'.format(ind))
        Draw.MolToFile(mol, filepath, size=(400,400),legend=s)
def return_csv(generated_all_smiles,path,old):
    import pandas as pd
    df1 = pd.DataFrame(columns = ['diversity',  'unique','nolvety'])
    df1.to_csv(path)
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
    data = pd.DataFrame([list])
    data.to_csv(path,mode='a',header=False,index=False)#
    return valid_mol

def pretrain(args,Prio,optimize,train_dat,epoc):
    """Trains the Prior RNN"""


    losses = CrossEntropyLoss()


    Prio.train()
    total_loss=0

    for step, batch in tqdm(enumerate(train_dat), total=len(train_dat)):

        # Sample from DataLoader
        prevs,next,lens= batch
        device=args.cuda
        prevs=prevs.to(device)
        next=next.to(device)
        lens=[l.to(device)for l in lens]
     
        
        outputs, _, _ = Prio(prevs, lens)
        


        # Calculate loss
        loss = losses(outputs.view(-1, outputs.shape[-1]),
                            next.view(-1))

        # Calculate gradients and take a step
        optimize.zero_grad()
        loss.backward()
        optimize.step()
        total_loss+=loss.item()

    return total_loss/step

def train_agents_charnn(input_param,args,
                restore_agent_from=None,
            
                scoring_function_kwargs=None,
                 learning_rate=0.001,
               n_steps=5000,
                
                ):

    voc = Vocabulary(init_from_file=args.vocab_path)



   
    Agent = CharRNN(voc,args)


    if args.gpu:
        device=args.cuda
       
        Agent=Agent.to(device)
    else:
        device='cpu'
     
        Agent=Agent.to(device)
    optimizer = torch.optim.Adam(Agent.parameters(), lr=args.lr)
    state = torch.load(restore_agent_from,map_location=device)
    # pretrain_state_dict = state['state_dict']
    pretrain_state_dict=state['state_dict']

    Agent.load_state_dict(pretrain_state_dict)



  
  

  


  

    # Information for the logger
    # step_score = [[], []]
    # # df1 = pd.DataFrame(columns = ['step',  'smiles','score','prior','agent','valid_smile'])
    # df1 = pd.DataFrame(columns = ['step',  'score','valid_smile'])
    # df1.to_csv(args.save_logs)

    print("Model initialized, starting training...")
    # raw_score=0
    # raw_smi=args.testmol
    with open(input_param['testfile'], "r") as f:
      old = f.readlines()
    old=[od[:-1] for od in old ]
    print(old)
    old=get_sanitize(old)

    locate=old.copy()
    
    val_ss=[]
    for step in range(n_steps):
        train_ss=[]
        

        # Sample from Agent
        seqs = Agent.sample(args.n_batch)
        smiles=reback(Agent,seqs,args.n_batch)
        score,scores=average_agg_tanimoto(fingerprints(old,n_jobs=1), fingerprints(smiles,n_jobs=1),device=device)
        val_ss.append(scores)

        if scores>=input_param['threshold']:
            new=Agent.sample(n_batch=input_param['n_batchs'])
            smiless=reback(Agent,new,input_param['n_batchs'])
            # smiless=get_newdata(smiless)
            a=return_csv(smiless,path=args.result_name,old=old)
            score,scores=average_agg_tanimoto(fingerprints(old,n_jobs=1), fingerprints(smiless,n_jobs=1),device=device)
            save_pngs(args,smiless,score)
            f=open(args.savefile,'w')
            for line in smiless:
                f.write(line+'\n')
            break
        smiles=[]
        smiles.extend(locate)
   



       
       
        dict1 = {'smiles': smiles} 
        df = pd.DataFrame(dict1)  
 
        df.to_csv(args.save_logs,mode='w')
        moldatatr=VocabDatasets(fname=args.save_logs,voc=voc)
        train_data = DataLoader(moldatatr, batch_size=1, shuffle=True, drop_last=True,
                      collate_fn=collate_fn)
        for i in tqdm(range(1), desc='Processing'):

            train_loss=pretrain(args,Prio=Agent,optimize=optimizer,train_dat=train_data,epoc=1)
          
 

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser("preprocess and train")
    group=add_optimizer_args(parser)

    
    # parsing.add_train_args(parser)
    # parser.add_argument_group
    args = parser.parse_args()

    with open('/home/chengkaiyang/Main/Script/charnn_Script/param.json', 'r') as f:

        args= json.load(f)
    with open('/home/chengkaiyang/Main/Script/charnn_Script/params.json', 'r') as f:

        input_param= json.load(f)
    args = argparse.Namespace(**args)
    # with open('/home/chengkaiyang/Main/Script/aae_script/param.json', 'r') as f:

    #     args= json.load(f)
    # with open('/home/chengkaiyang/Main/Script/aae_script/params.json', 'r') as f:

    #     input_param= json.load(f)



    train_agents_charnn(input_param=input_param,args=args,restore_agent_from=args.restore_agent_from,scoring_function_kwargs={})
