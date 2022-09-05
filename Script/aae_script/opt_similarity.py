'''
Descripttion: 
version: 
Author: 成凯阳
Date: 2022-05-07 08:31:38
LastEditors: 成凯阳
LastEditTime: 2022-06-20 18:40:21
'''


#!/usr/bin/env python



# from sklearn import metrics

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm
import numpy as np
from Utils.utils.data_utils import fp_print
from Dataset.get_Vocab import AAEVocabDatasets,Vocabulary,collate_fnS
from Model.aae_model import AAE
import argparse
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
from torch import  optim
import pandas as pd
import os
from rdkit import rdBase
from rdkit.Chem import AllChem,MACCSkeys,Descriptors
from rdkit import DataStructs
rdBase.DisableLog('rdApp.error')

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
    group.add_argument("--beta1", help="Adam beta 1", type=float, default=0.9)
    group.add_argument("--beta2", help="Adam beta 2", type=float, default=0.998)
    group.add_argument("--eps", help="Adam epsilon", type=float, default=1e-9)
    group.add_argument("--weight_decay", help="Adam weight decay", type=float, default=1e-4)
    group.add_argument("--save_logs", help="path to save test smiles", type=str, default='/home/chengkaiyang/Main/opt/aae/optmetrics.csv')
    group.add_argument("--gpu", help="use gpu", type= bool, default=True)
    group.add_argument("--max_length", help="max length to sample", type=int, default=100)
    group.add_argument("--n_batch", help="batch size to sample", type=int, default=200)
    group.add_argument("--num_layers", help="Model layers", type=int, default="2")
    # group.add_argument("--train_p", help="Checkpoint to load", type=str, default='trainfilter.csv')
    group.add_argument("--save_logss", help="path to save metric", type=str, default='/home/chengkaiyang/Main/opt/aae/metric.csv')
    group.add_argument("--save_dir", help="save dir include all files", type=str, default='/home/chengkaiyang/Main/opt/aae/')
    group.add_argument("--dropout", help="random sample point ", type=float, default="0.")
    group.add_argument("--restore_agent_from", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/savehuizong/aae/aaemodels.20.pt')
    group.add_argument("--vocab_path", help="Vocab path to load", type=str, default='/home/chengkaiyang/Main/datanew/data/Voc.txt')

    group.add_argument("--testmol", help="test mol similarity ", type=str, default='CC1CC(NBr)C=C1N=CC(=O)CN(C)O')
    group.add_argument("--cuda", help="use gpu device", type=str, default='cuda:0')
    group.add_argument("--threshold", help="similarity threshold", type=float, default=0.7)
   

  
    return group


def reback(model,toens,n_batch):
    smis=[]
   

    for i in range(n_batch):
        stri=''
        for j in range(len(toens[i].cuda().cpu().numpy())):
            smi=model.vocabulary.reversed_vocab[int(toens[i][j])]
            stri+=smi
        stri=stri.split('<pad>')[0]
        if Chem.MolFromSmiles(stri):
            smis.append(stri)

        # smis.append(stri)
       

        

    return smis
def pretrain(args,Prior,autooptimizer,disoptimizer,train_dat,epoc):
    """Trains the Prior RNN"""

    # Read vocabulary from a file
    # voc = Vocabulary(init_from_file="Data/vocab.txt")
    losses = CrossEntropyLoss()
    dloss=BCEWithLogitsLoss()

    Prior.train()
    total_losses=0
    gloss=0
    disloss=0

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
    return total_losses

def train_agents(query_fp,args,
                restore_agent_from=None,
                
                scoring_function_kwargs=None,
                save_dir=None, learning_rate=0.0005,
                batch_size=None, n_steps=700,
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
        lr=args.lr,
    
        weight_decay=args.weight_decay
    )
 
    doptimizer = optim.Adam(
        Agent.discriminator.parameters(),
        lr=args.lr,
     
        weight_decay=args.weight_decay
    )
  
    state = torch.load(restore_agent_from,map_location=device)
    # pretrain_state_dict = state['state_dict']
    pretrain_state_dict=state['state_dict']

    Agent.load_state_dict(pretrain_state_dict)

    df1 = pd.DataFrame(columns = ['smiles', 'best score','avg score'])
    df1.to_csv(args.save_logss)
    raw_score=0
    raw_smi=args.testmol

    print("Model initialized, starting training...")
    old_smi=[]

    for step in range(n_steps):
        train_ss=[]
        val_ss=[]

        # Sample from Agent
        new=Agent.sample(n_batch=args.n_batch,args=args)
    # new_pad=pad_sequence(new,batch_first=True)
    
        smiles=reback(Agent,new,args.n_batch)
        score=fp_print(smiles,query_fp)
        # score=pair_log(smiles)
   
  
        # score=fp_print(smiles,query_fp)
        s=np.mean(score)
        if s>args.threshold:
            print('get threshold')
            break
        max_score=score.index(max(score))
        max_smiles=smiles[max_score]
        idx=[i for i,a in enumerate(score) if a >s]
        # idx = heapq.nlargest(10, range(len( score)),  score.__getitem__)
        smiles=[smiles[id]  for id in idx]
        old_smi=smiles
        # for i in smiles:
        #     if i not in old_smi:
        #         old_smi.append(i)
        if max_score>raw_score:
            old_smi.append(max_smiles)
            old_smi.append(args.testmol)
            old_smi.append(args.testmol)
            # old_smi.append(args.testmol)
            # smiles.remove(min_smiles)
        else:
            old_smi.append(raw_smi)
            old_smi.append(args.testmol)
            old_smi.append(args.testmol)
            old_smi.append(args.testmol)
        # if max_score>raw_score:
        #     smiles.append(max_smiles)
        #     smiles.append(args.testmol)
        #     smiles.append(args.testmol)
        #     # smiles.remove(min_smiles)
        # else:
        #     smiles.append(raw_smi)
        #     smiles.append(args.testmol)
        #     smiles.append(args.testmol)
      
       
       
        dict1 = {'smiles': old_smi} 
        df = pd.DataFrame(dict1)  
 
        df.to_csv(args.save_logs,mode='w')
        moldatatr=AAEVocabDatasets(fname=args.save_logs,voc=voc)
     
  
        train_data = DataLoader(moldatatr, batch_size=10, shuffle=True, drop_last=True,
                      collate_fn=collate_fnS)
   
        for i in tqdm(range(1), desc='Processing'):
            train_loss=pretrain(args,Prior=Agent,autooptimizer=aoptimizer,disoptimizer=doptimizer,train_dat=train_data,epoc=i)

      
  
        lists = [max_smiles, max(score), s]
        data = pd.DataFrame([lists])
        data.to_csv(args.save_logss,mode='a',header=False,index=False)#
        torch.save(Agent.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))
        raw_smi=max_smiles
        raw_score=max(score)
   

  
  
    # torch.save(Agent.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser("optimizering options")
    group=add_optimizer_args(parser)

    
 
    args = parser.parse_args()



    train_agents(query_fp=args.testmol,args=args,save_dir=args.save_dir,restore_agent_from=args.restore_agent_from,scoring_function_kwargs={})
