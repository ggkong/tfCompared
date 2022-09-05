'''
Descripttion: 
version: 
Author: 成凯阳
Date: 2022-05-07 08:31:38
LastEditors: 成凯阳
LastEditTime: 2022-08-31 05:48:07
'''


#!/usr/bin/env python



# from sklearn import metrics

from ast import Break
from multiprocessing.spawn import old_main_modules
import sys, os

# from Utils.torch_jtnn.chemutils import count

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
import torch.nn as nn
# from step5_opt_sim import train_agents
from Utils.utils.train_utils import NoamLR,decrease_learning_rate
from Utils.utils.metric import fraction_valid_smiles
from torch.nn import CrossEntropyLoss
from torch import log_, optim
import pandas as pd
import random
import os
from rdkit.Chem import Draw
from Utils.torch_jtnn.chemutils import get_sanitize
from rdkit import rdBase
from rdkit.Chem import AllChem,MACCSkeys,Descriptors
from rdkit import DataStructs
rdBase.DisableLog('rdApp.error')
def pretrain(args,Prior,autooptimizer,disoptimizer,train_dat,epoc):
    """Trains the Prior RNN"""

    # Read vocabulary from a file
    # voc = Vocabulary(init_from_file="Data/vocab.txt")
    losses = CrossEntropyLoss()
    dloss=nn.BCEWithLogitsLoss()

  

    # optimizer = torch.optim.Adam(Prior.parameters(), lr = 0.002)
    # for epoch in range(1, 6):
    Prior.train()
    total_losses=0
    gloss=0
    disloss=0
    # df1 = pd.DataFrame(columns = ['epoch', 'step', 'loss'])
    # df1.to_csv("/user-data/Main/logs/train_metrics.csv")
    # When training on a few million compounds, this model converges
    # in a few of epochs or even faster. If model sized is increased
    # its probably a good idea to check loss against an external set of
    # validation SMILES to make sure we dont overfit too much.
    for i, batch in tqdm(enumerate(train_dat), total=len(train_dat)):
        device = torch.device(args.cuda)
      

     # Sample from DataLoader
        encoder_inputs,decoder_inputs,decoder_targets= batch
        x,lengths=encoder_inputs
        x=x.to(device)
        lengths=torch.Tensor(lengths).to(device)
        y,lengthsnew=decoder_inputs
        y=y.to(device)
        lengthsnew=torch.Tensor(lengthsnew).to(device)
     

        
        
        outputs= Prior.encoder(x,lengths)
        decoder_outputs, decoder_output_lengths, _ = Prior.decoder(
                y,lengthsnew,outputs, is_latent_states=True)
        logits = torch.softmax(decoder_outputs, -1)
        logits = logits.contiguous().view(-1, logits.shape[-1])
        currents = torch.distributions.Categorical(logits).sample()
        discriminator_outputs = Prior.discriminator(outputs)
        decoder_outputs = torch.cat(
                [t[:l] for t, l in zip(decoder_outputs,
                                       decoder_output_lengths)], dim=0)
        # decoder_output =[t[:l] for t, l in zip(decoder_outputs,
        #                                decoder_output_lengths)]
        decoder_targets = torch.cat(
                [t[:l] for t, l in zip(*decoder_targets)], dim=0)
        # decoder_target = [t[:l] for t, l in zip(*decoder_targets)]
        decoder_targets=decoder_targets.to(device)
        
        if i % (args.discriminator_steps + 1) == 0:



            autoencoder_loss =losses(
                    decoder_outputs, decoder_targets
                )
            discriminator_targets = torch.ones(
                     outputs.shape[0], 1, device=device
                 )
            generator_loss = dloss(
                    discriminator_outputs, discriminator_targets
             )
            total_loss = autoencoder_loss + generator_loss
            gloss+=autoencoder_loss.item()
            total_losses+=total_loss.item()
            
        else:

            discriminator_targets = torch.zeros(
                    outputs.shape[0], 1, device=device
                )
            generator_loss = dloss(
                    discriminator_outputs, discriminator_targets
                )

            discriminator_inputs =torch.randn(outputs.shape[0], args.latent_size).to(device)
           
            discriminator_outputs = Prior.discriminator(
                    discriminator_inputs
                )
            discriminator_targets = torch.ones(
                    outputs.shape[0], 1, device=device
                )
            discriminator_loss = dloss(
                    discriminator_outputs, discriminator_targets
                )
            total_loss = 0.5*generator_loss + 0.5*discriminator_loss
            disloss+=discriminator_loss.item()
            # gloss+=generator_loss.item()
            total_losses+=total_loss.item()
        autooptimizer.zero_grad()
        disoptimizer.zero_grad()
        total_loss.backward()
        # for parameter in Prior.parameters():

        #     parameter.grad.clamp_(-5, 5)
        # nn.utils.clip_grad_norm_(Prior.parameters(), max_norm=5)
        if i % (args.discriminator_steps + 1) == 0:
            autooptimizer.step()
        else:

            disoptimizer.step()
   

            
        


    
        total_losses+=total_loss.item()
        # disloss+=discriminator_loss.item()
        # gloss+=generator_loss.item()
    return total_losses
# import re

# from Utils.utils.scorefunction import pair_log

# from rdkit.Chem import Draw
# from joblib import Parallel,delayed
# from torch.nn.utils.rnn import pad_sequence
# from collections import Counter
# from Utils.utils.data_utils import Variable
# from rdkit.Chem import MolFromSmiles


   

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
    group.add_argument("--learningrate", help="Learning rate", type=float, default=1e-4)
    group.add_argument("--beta1", help="Adam beta 1", type=float, default=0.9)
    group.add_argument("--beta2", help="Adam beta 2", type=float, default=0.998)
    group.add_argument("--eps", help="Adam epsilon", type=float, default=1e-9)
    group.add_argument("--weight_decay", help="Adam weight decay", type=float, default=1e-4)
    group.add_argument("--save_logs", help="path to save logs", type=str, default='/home/chengkaiyang/new/opt/aae/optmetrics.csv')
    group.add_argument("--gpu", help="use gpu", type= int, default='1')
    group.add_argument("--max_length", help="max length to sample", type=int, default=100)
    group.add_argument("--n_batch", help="batch size to sample", type=int, default=10000)
    group.add_argument("--num_layers", help="Model layers", type=int, default="2")
    # group.add_argument("--train_p", help="Checkpoint to load", type=str, default='trainfilter.csv')
    group.add_argument("--save_logss", help="path to save check", type=str, default='/home/chengkaiyang/new/opt/aae/metric.csv')
    group.add_argument("--save_dir", help="path to save check", type=str, default='/home/chengkaiyang/Main/opt/aae/')
    group.add_argument("--dropout", help="random sample point ", type=float, default="0.")
    group.add_argument("--restore_agent_from", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/opt/aae/Agent.ckpt')
    group.add_argument("--vocab_path", help="Vocab path to load", type=str, default='/home/chengkaiyang/Main/datanew/data/Voc.txt')
    group.add_argument("--restore_prior_from", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/savehuizong/aae/aaemodels.20.pt')
    group.add_argument("--testmol", help="test mol similarity ", type=str, default='Oc(cc1c(nc2)c(F)c3c2c(N4CC5CCC(N5)C4)nc(OC[C@@]67CCCN6C[C@@H](C7)F)n3)cc8c1c(C#C)c(F)cc8')
    group.add_argument("--cuda", help="use gpu device", type=str, default='cuda:1')
    group.add_argument("--threshold", help="similarity threshold", type=float, default=0.8)
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
def geiding(old):
    if len(old)>100:
        return 10000
    else:
        return 200


def pro(mol):
    return Chem.MolFromSmiles(mol)
def save_pngs(args,x_smi,score):
    for ind,(i,s) in enumerate(zip(x_smi,score)):

        mol=Chem.MolFromSmiles(i)
        lista = [i]
        datass = pd.DataFrame([lista])
        datass.to_csv('./aae_script/renwuyi.csv',mode='a',header=False,index=False)#
        data_di=args.tmp_dir
        s = ' {:.2f}'.format(s)
        filepath=os.path.join(data_di, '{}.png'.format(ind))
        Draw.MolToFile(mol, filepath, size=(400,400),legend=s)
def train_agent_opt(input_param,query_fp,args,
                restore_agent_from=None,
            
                scoring_function_kwargs=None,
                 learning_rate=0.0005,
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
    # df1s.to_csv('/home/chengkaiyang/Main/Script/aae_script/3.csv')

    print("Model initialized, starting training...")
    old=[]
    cs=0

    for step in range(n_steps):
        train_ss=[]
        val_ss=[]
        batchs=geiding(old)
        print(batchs)

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
            if i not in old:
                s_i=similarity(i, query_fp)
                old.append(i)
                lista = [i]
                datass = pd.DataFrame([lista])
                datass.to_csv('../Script/3.csv',mode='a',header=False,index=False)#
                data_di=args.tmp_dir
                s_i = ' {:.2f}'.format(s_i)
                filepath=os.path.join(data_di, '{}.png'.format(cs))
                cs+=1
                mos=pro(i)
                Draw.MolToFile(mos, filepath, size=(400,400),legend=s_i)
                if len(old)>=input_param['n_batchs']:
                    
                    # print('Load trained model and generate data done! Time {:.2f} seconds'.format(time.time() - start))
                    sys.exit(0)
        # break
                
       
        smiles=[]
        
                    #     from Utils.torch_jtnn.chemutils import get_sanitize,decode_stereo
        #     choice_ls=[0,1]
        #     print('Load trained model and generate data done! Time {:.2f} seconds'.format(time.time() - start))
        #     f=open('/home/chengkaiyang/Main/Script/aae_script/gen_su.txt',"w")
        #     for line in old:
        #         random_choice = np.random.choice(choice_ls, 1)[0]
        #         if random_choice ==1:
        #             line=decode_stereo(line)
        #             line=line[0]
        #         else:
        #             line=line

       
        if len(old)>3000:
            old_new=random.sample(old, 3000)
        elif len(old) <75:
            old_new=random.sample(old, int(1/3*len(old)))
        elif len(old) <50:
            old_new=random.sample(old, int(1/3*len(old)))
        elif len(old) <25:
            old_new=random.sample(old, int(1/4*len(old)))
        elif len(old) <12:
            old_new=random.sample(old, int(1/3*len(old)))
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
        smiles=panduan(smiles,voc)
    
           
      
       
       
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
        if len(smiles)>100:
            bat=int(1/4*len(smiles))
        else:
            bat=1
     
  
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
        # if len(old)>=1:
        #     from Utils.torch_jtnn.chemutils import get_sanitize,decode_stereo
        #     choice_ls=[0,1]
        #     print('Load trained model and generate data done! Time {:.2f} seconds'.format(time.time() - start))
        #     f=open('/home/chengkaiyang/Main/Script/aae_script/gen_su.txt',"w")
        #     for line in old:
        #         random_choice = np.random.choice(choice_ls, 1)[0]
        #         if random_choice ==1:
        #             line=decode_stereo(line)
        #             line=line[0]
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
    import time,json
    start = time.time()
    file_path=os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(file_path,'params.json'), 'r') as f:

        input_param= json.load(f)
    print("Start at Time: {}".format(time.ctime()))
    #train_agents(query_fp=args.testmol,args=args,save_dir=args.save_dir,restore_agent_from=args.restore_prior_from,scoring_function_kwargs={})



    train_agent_opt(input_param=input_param,query_fp=args.testmol,args=args,restore_agent_from=args.restore_prior_from,scoring_function_kwargs={})
