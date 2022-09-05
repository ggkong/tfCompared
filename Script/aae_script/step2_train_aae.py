'''
Author: 成凯阳
Date: 2022-03-19 09:30:03
LastEditors: 成凯阳
LastEditTime: 2022-06-17 01:28:26
FilePath: /Main/Script/aae_script/step2_train_aae.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''

#!/usr/bin/env python
from pickletools import optimize
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from Dataset.get_dataset import get_dataset,get_lookup_tables
from Dataset.get_Vocab import AAEVocabDatasets,Vocabulary, collate_fnS
from Model.aae_model import AAE
import argparse
# from Utils.utils import parsing
from Utils.utils.train_utils import NoamLR,decrease_learning_rate
from torch.nn import CrossEntropyLoss
from torch import optim
import pandas as pd
import os
import torch.nn as nn
# from utils import Variable, decrease_learning_rate
rdBase.DisableLog('rdApp.error')


   


def add_train_args(parser):
    group = parser.add_argument_group("Training options")
#     # file paths
# 
    group.add_argument("--restore", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/savehuizong/aae/aaemodels.2.pt')
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
    
  
 
    group.add_argument("--train_p", help="train dataset", type=str, default='/home/chengkaiyang/Main/datanew/data/train.csv')
    group.add_argument("--valid_p", help="valid dataset", type=str, default='/home/chengkaiyang/Main/datanew/data/val.csv')
    group.add_argument("--test_p", help="valid dataset", type=str, default='/home/chengkaiyang/Main/data/data/test.csv')
    group.add_argument("--save_log", help="metric to save", type=str, default='/home/chengkaiyang/Main/logshuizong/aaenewmetrics.csv')
    group.add_argument("--vocab_path", help="Vocab path to load", type=str, default='/home/chengkaiyang/Main/datanew/data/Voc.txt')
    group.add_argument("--hidden", help="Model hidden size", type=int, default="128")
    group.add_argument("--num_layers", help="Model layers", type=int, default="2")
    group.add_argument("--dropout", help="random sample point ", type=float, default="0.")
    group.add_argument("--train_batch_size", help="batch_size", type=int, default="2600")
    group.add_argument("--valid_batch_size", help="batch_size", type=int, default="1024")
  
    group.add_argument("--save_dir", help="path to save ", type=str, default="/home/chengkaiyang/Main/savehuizong/aae")
    group.add_argument("--lr", help="Learning rate", type=float, default=0.01)
    group.add_argument("--beta1", help="Adam beta 1", type=float, default=0.9)
    group.add_argument("--beta2", help="Adam beta 2", type=float, default=0.998)
    group.add_argument("--eps", help="Adam epsilon", type=float, default=1e-9)
    group.add_argument("--weight_decay", help="Adam weight decay", type=float, default=0)
    group.add_argument("--warm_up", help="Adam weight decay", type=int, default=50)
    group.add_argument("--gpu", help="use gpu or not", type=str, default='True')
    group.add_argument("--cuda", help="use gpu device", type=str, default='cuda:0')

   
    return group
def rebacksmiles(args,Prior,autooptimizer,disoptimizer,train_dat):
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
        lie=[]
        decoder_output = [t[:l] for t, l in zip(decoder_outputs,
                                       decoder_output_lengths)]
        decoder_target = [t[:l] for t, l in zip(*decoder_targets)]
        for d in decoder_output:
            smi=''
            for j in d:
                logits = torch.softmax(j,-1)
                currents = torch.distributions.Categorical(logits).sample()
                token=Prior.vocabulary.reversed_vocab[int(currents)]
                if token=='<eos>':


                    break
                smi+=token
                
                
        #     lie.append(smi)


        # print(decoder_output)

        # for i in decoder_outputs:
        #     smi=''
            
        #     for j in i:
             

        #         logits = torch.softmax(j, -1)
        #         # logits = logits.contiguous().view(-1, logits.shape[-1])
        #         currents = torch.distributions.Categorical(logits).sample()
        #         token=Prior.vocabulary.reversed_vocab[int(currents)]
        #         if token=='<eos>':
        #             break
        #         smi+=token
                
                
        #     lie.append(smi)







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
    return total_losses/i,gloss/i,disloss/i
        # flag=step % 20 == 0 


        # Every 500 steps we decrease learning rate and print some information
        # if flag and step != 0:
        #     # decrease_learning_rate(optimizer, decrease_by=0.001)
        #     tqdm.write("*" * 50)
        #     tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.item()))

        #     list = [epoch, step, loss.item()]

        #     data = pd.DataFrame([list])
        #     data.to_csv('/user-data/Main/logs/train_metrics.csv',mode='a',header=False,index=False)#

                # seqs, likelihood, _ = Prior.sample(128)
                # valid = 0
                # for i, seq in enumerate(seqs.cpu().numpy()):
                #     smile = voc.decode(seq)
                #     if Chem.MolFromSmiles(smile):
                #         valid += 1
                #     if i < 5:
                #         tqdm.write(smile)
                # tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                # tqdm.write("*" * 50 + "\n")
                # state = {
                #     "args": args,
                #     "state_dict": Prior.state_dict()
                # }

                # torch.save(state, os.path.join(args.save_dir, f"model.{epoch}_{step}.pt"))

        # Save the Prior
def run_an_eval_epoch(args,Prior,valid_dat,epoc):
    """Trains the Prior RNN"""

    # Read vocabulary from a file
    # voc = Vocabulary(init_from_file="Data/vocab.txt")
    losses = CrossEntropyLoss()
    dloss=nn.BCEWithLogitsLoss()

  

    # optimizer = torch.optim.Adam(Prior.parameters(), lr = 0.002)
    # for epoch in range(1, 6):
    Prior.eval()
    total_losses=0
    gloss=0
    disloss=0
    with torch.no_grad():

        for i, batch in tqdm(enumerate(valid_dat), total=len(valid_dat)):


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
            discriminator_outputs = Prior.discriminator(outputs)
            decoder_outputs = torch.cat(
                [t[:l] for t, l in zip(decoder_outputs,
                                       decoder_output_lengths)], dim=0)
            decoder_targets = torch.cat(
                [t[:l] for t, l in zip(*decoder_targets)], dim=0)
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
                # gloss+=generator_loss.item()
                disloss+=discriminator_loss.item()
                total_losses+=total_loss.item()
    #         if i%5==0:


    #             new=Prior.sample(n_batch=100,args=args)
                
    # # new=Prior.sample(n_batch=args.n_batch,max_length=args.max_length)
    #             d=reback(Prior,new,100)
    #             d=[smile for smile in d if Chem.MolFromSmiles(smile)]
    #             with open(os.path.join('/home/chengkaiyang/Main/Script/aae_script', "validsampled"), 'a') as f:

    #                 f.write("{}\n".format(len(d)))
      
              



            
        


    
            # total_losses+=total_loss.item()
        return total_losses/i,gloss/i,disloss/i

    # with no grad
    # df1 = pd.DataFrame(columns = ['epoch', 'step', 'loss'])
    # df1.to_csv("/user-data/Main/logs/valid_metrics.csv")
    # for step, batch in tqdm(enumerate(valid_data), total=len(valid_data)):

    #     # Sample from DataLoader
    #     prevs,next,lens= batch
        
    #     outputs, _, _ = Prior(prevs, lens)
    #     loss = losses(outputs.view(-1, outputs.shape[-1]),
    #                         next.view(-1))
    #     to+=loss.item()
    
        # flag=step % 10 == 0
        # if flag and step != 0:
        #     # decrease_learning_rate(optimizer, decrease_by=0.001)
        #     tqdm.write("*" * 50)
        #     tqdm.write("Epoch {:3d}   step {:3d} valid  loss: {:5.2f}\n".format(epoch, step, loss.item()))
        #     state = {
              
        #         "state_dict": Prior.state_dict()
        #     }

        #     torch.save(state, os.path.join(args.save_dir, f"model.{epoch}_{step}.pt"))
        #     list = [epoch, step, loss.item()]

        #     data = pd.DataFrame([list])
        #     data.to_csv('/user-data/Main/logs/valid_metrics.csv',mode='a',header=False,index=False)#
      
        
def reback(model,toens,n_batch):
    smis=[]
   

    for i in range(n_batch):
        stri=''
        for j in range(len(toens[i].cuda().cpu().numpy())):
            smi=model.vocabulary.reversed_vocab[int(toens[i][j])]
            stri+=smi
        stri=stri.split('<pad>')[0]
        smis.append(stri)
       

        

    return smis 
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("preprocess and train")
    group=add_train_args(parser)
    

    
    # parsing.add_train_args(parser)
    # parser.add_argument_group
   
    args = parser.parse_args()
    
    device=args.cuda
    voc = Vocabulary(init_from_file=args.vocab_path)
    # restore_from='1.pt'
    # df1 = pd.DataFrame(columns = ['epoch',  'trainloss','validloss','correct'])
    df1 = pd.DataFrame(columns = ['epoch',  'trainloss','validloss','testloss','traingen','validgen','testgen','traindis','validdis','testdis'])
    df1.to_csv(args.save_log)
    # df2 = pd.DataFrame(columns = ['epoch', 'step', 'loss'])
    # df2.to_csv("/user-data/Main/logs/valid_metrics.csv")
    

    # Create a Dataset from a SMILES file

    moldatatr=AAEVocabDatasets(fname=args.train_p,voc=voc)
    moldatate=AAEVocabDatasets(fname=args.valid_p,voc=voc)
    # moldatace=AAEVocabDatasets(fname=args.test_p,voc=voc)
 
    train_data = DataLoader(moldatatr, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                      collate_fn=collate_fnS)
    valid_data = DataLoader(moldatate, batch_size=args.valid_batch_size, shuffle=True, drop_last=True,
                      collate_fn=collate_fnS)
    # test_data = DataLoader(moldatace, batch_size=args.valid_batch_size, shuffle=True, drop_last=True,
    #                   collate_fn=collate_fnS)

    Prior = AAE(voc,args)
    if args.gpu:
        Prior=Prior.to(device)
    else:
        Prior=Prior


    # Can restore from a saved RNN
    if args.restore=='':
        print("start new training ...")
        pass
    

 
    else:
        
      

        
        state = torch.load(args.restore)
        pretrain_state_dict = state['state_dict']
        Prior.load_state_dict(pretrain_state_dict)
        print("restore from {} ...".format(args.restore))
    aoptimizer = optim.AdamW(
          list(Prior.encoder.parameters()) +
                list(Prior.decoder.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    # aoptimizer=torch.optim.RMSprop(list(Prior.encoder.parameters()) +
    #             list(Prior.decoder.parameters()), lr=0.05, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
 
    doptimizer = optim.AdamW(
        Prior.discriminator.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
 
    # doptimizer=torch.optim.RMSprop(Prior.discriminator.parameters(), lr=0.05, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    
    # ascheduler = NoamLR(
    #     aoptimizer,
    #     model_size=128,
    #     warmup_steps=args.warm_up
    # )
    # dscheduler = NoamLR(
    #     doptimizer,
    #     model_size=128,
    #     warmup_steps=args.warm_up
    # )
    #rebacksmiles(args,Prior=Prior,autooptimizer=aoptimizer,disoptimizer=doptimizer,train_dat=train_data)
    for epoch in range(1, 300):
        train_loss,g,d=pretrain(args,Prior=Prior,autooptimizer=aoptimizer,disoptimizer=doptimizer,train_dat=train_data,epoc=epoch)
        valid_loss,tg,td=run_an_eval_epoch(args,Prior=Prior,valid_dat=valid_data,epoc=epoch)
 
        t_loss,vg,vd=run_an_eval_epoch(args,Prior=Prior,valid_dat=valid_data,epoc=epoch)
        
        # scheduler.step()
        
        list = [epoch,  train_loss, valid_loss,t_loss,g,tg,vg,d,td,vd]
        data = pd.DataFrame([list])
        data.to_csv(args.save_log,mode='a',header=False,index=False)#

      
        state = {
              
                "state_dict": Prior.state_dict()
            }

        torch.save(state, os.path.join(args.save_dir, f"aaemodels.{epoch}.pt"))

