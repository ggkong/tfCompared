'''
Author: 成凯阳
Date: 2022-03-11 15:22:25
LastEditors: 成凯阳
LastEditTime: 2022-06-17 13:17:37
FilePath: /Main/Script/charnn_Script/step2_train_charnn.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
#!/usr/bin/env python

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from Dataset.get_dataset import get_dataset,get_lookup_tables

from Dataset.get_Vocab import VocabDatasets,collate_fn,Vocabulary
from Model.model import CharRNN
import argparse
# from Utils.utils import parsing
from Utils.utils.train_utils import NoamLR,decrease_learning_rate
from torch.nn import CrossEntropyLoss
from torch import optim
import pandas as pd
from step3_sample_charnn import reback
# from utils import Variable, decrease_learning_rate
rdBase.DisableLog('rdApp.error')


   


def add_train_args(parser):
    group = parser.add_argument_group("Training options")
#     # file paths
# 
    group.add_argument("--restore", help="Checkpoint to load", type=str, default='/home/chengkaiyang/Main/savehuizong/charnn/model.490.pt')
  
 
    group.add_argument("--train_p", help="train dataset", type=str, default='/home/chengkaiyang/Main/datanew/data/424.csv')
    group.add_argument("--valid_p", help="valid dataset", type=str, default='/home/chengkaiyang/Main/datanew/data/val.csv')
    group.add_argument("--test_p", help="valid dataset", type=str, default='/home/chengkaiyang/Main/data/data/test.csv')
    group.add_argument("--save_log", help="metric to save", type=str, default='/home/chengkaiyang/Main/logshuizong/charmetrics.csv')
    group.add_argument("--vocab_path", help="Vocab path to load", type=str, default='/home/chengkaiyang/Main/datanew/data/Voc.txt')
    group.add_argument("--hidden", help="Model hidden size", type=int, default="256")
    group.add_argument("--epoch", help="train epoch", type=int, default="500")
    group.add_argument("--num_layers", help="Model layers", type=int, default="3")
    group.add_argument("--dropout", help="random sample point ", type=float, default="0.1")
    group.add_argument("--train_batch_size", help="batch_size", type=int, default="10000")
    group.add_argument("--valid_batch_size", help="batch_size", type=int, default="1024")
  
    group.add_argument("--save_dir", help="path to save ", type=str, default="/home/chengkaiyang/Main/savehuizong/charnn/")
    group.add_argument("--lr", help="Learning rate", type=float, default=0.005)
    group.add_argument("--beta1", help="Adam beta 1", type=float, default=0.9)
    group.add_argument("--beta2", help="Adam beta 2", type=float, default=0.998)
    group.add_argument("--eps", help="Adam epsilon", type=float, default=1e-9)
    group.add_argument("--weight_decay", help="Adam weight decay", type=float, default=1e-2)
    # group.add_argument("--warm_up", help="Adam weight decay", type=int, default=1)
    group.add_argument("--gpu", help="use gpu or not", type=bool, default='True')
    group.add_argument("--cuda", help="use gpu device", type=str, default='cuda:1')
   
    return group
def pretrain(args,Prio,optimize,train_dat,epoc):
    """Trains the Prior RNN"""

    # Read vocabulary from a file
    # voc = Vocabulary(init_from_file="Data/vocab.txt")
    losses = CrossEntropyLoss()

    # Create a Dataset from a SMILES file
    # path1='/user-data/Main/Data/1.csv'
    # moldata=VocabDatasets(fname=path1,voc=voc)
 
    # train_data = DataLoader(moldata, batch_size=2, shuffle=True, drop_last=True,
    #                   collate_fn=collate_fn)
    # valid_data = DataLoader(moldata, batch_size=2, shuffle=True, drop_last=True,
    #                   collate_fn=collate_fn)

    # Prior = CharRNN(voc,args)
    # if args.gpu:
    #     Prior=Prior.cuda()
    # else:
    #     Prior=Prior


    # # Can restore from a saved RNN
    # if restore_from:
    #     Prior.rnn.load_state_dict(torch.load(restore_from))
    # optimizer = optim.AdamW(
    #     Prior.parameters(),
    #     lr=args.lr,
    #     betas=(args.beta1, args.beta2),
    #     eps=args.eps,
    #     weight_decay=args.weight_decay
    # )
    # scheduler = NoamLR(
    #     optimizer,
    #     model_size=128,
    #     warmup_steps=args.warm_up
    # )

    # optimizer = torch.optim.Adam(Prior.parameters(), lr = 0.002)
    # for epoch in range(1, 6):
    Prio.train()
    total_loss=0
    # df1 = pd.DataFrame(columns = ['epoch', 'step', 'loss'])
    # df1.to_csv("/user-data/Main/logs/train_metrics.csv")
    # When training on a few million compounds, this model converges
    # in a few of epochs or even faster. If model sized is increased
    # its probably a good idea to check loss against an external set of
    # validation SMILES to make sure we dont overfit too much.
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
    #     if step%5==0:


    #         new=Prior.sample(n_batch=100)
                
    # # new=Prior.sample(n_batch=args.n_batch,max_length=args.max_length)
    #         d=reback(Prior,new,100)
    #         d=[smile for smile in d if Chem.MolFromSmiles(smile)]
    #         with open(os.path.join('/home/chengkaiyang/Main/Script/charnn_Script', "validsampled"), 'a') as f:

    #             f.write("{}\n".format(len(d)))
    return total_loss/step
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
def run_an_eval_epoch(args,Prior,valid_data,epoch):
    losses = CrossEntropyLoss()
    
    Prior.eval()
    to=0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(valid_data), total=len(valid_data)):

          


            prevs,next,lens= batch
            device=args.cuda
            prevs=prevs.to(device)
            next=next.to(device)
            lens=[l.to(device)for l in lens]
        
            outputs, _, _ = Prior(prevs, lens)
            loss = losses(outputs.view(-1, outputs.shape[-1]),
                            next.view(-1))
            to+=loss.item()
        return to/step

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
      
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser("preprocess and train")
    group=add_train_args(parser)

    
    # parsing.add_train_args(parser)
    # parser.add_argument_group
   
    args = parser.parse_args()
    voc = Vocabulary(init_from_file=args.vocab_path)
    device=args.cuda
    # restore_from='1.pt'
    df1 = pd.DataFrame(columns = ['epoch',  'trainloss','validloss'])
    df1.to_csv(args.save_log)
    # df2 = pd.DataFrame(columns = ['epoch', 'step', 'loss'])
    # df2.to_csv("/user-data/Main/logs/valid_metrics.csv")
    

    # Create a Dataset from a SMILES file

    moldatatr=VocabDatasets(fname=args.train_p,voc=voc)
    moldatate=VocabDatasets(fname=args.valid_p,voc=voc)
 
    train_data = DataLoader(moldatatr, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
                      collate_fn=collate_fn)
    valid_data = DataLoader(moldatate, batch_size=args.valid_batch_size, shuffle=True, drop_last=True,
                      collate_fn=collate_fn)

    Prior = CharRNN(voc,args)
    if args.gpu:
        Prior=Prior.to(device)
    else:
        Prior=Prior


    # Can restore from a saved RNN
    if args.restore=='':
        print("start new training ...")
        # pass
    

 
    else:
        
      

        
        state = torch.load(args.restore)
        pretrain_state_dict = state['state_dict']
        Prior.load_state_dict(pretrain_state_dict)
        print("restore from {} ...".format(args.restore))
    # optimizer = optim.AdamW(
    #     Prior.parameters(),
    #     lr=args.lr,
    #     betas=(args.beta1, args.beta2),
    #     eps=args.eps,
    #     weight_decay=args.weight_decay
    # )
    # scheduler = NoamLR(
    #     optimizer,
    #     model_size=128,
    #     warmup_steps=args.warm_up
    # )
    optimizer = torch.optim.Adam(Prior.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        train_loss=pretrain(args=args,Prio=Prior,optimize=optimizer,train_dat=train_data,epoc=epoch)
        valid_loss=run_an_eval_epoch(args, Prior, valid_data=valid_data, epoch=epoch)
        lists = [epoch,  train_loss, valid_loss]
        # print(lists[0],lists[2])
        data = pd.DataFrame([lists])
        data.to_csv(args.save_log,mode='a',header=False,index=False)#

            # data = pd.DataFrame([list])
            # data.to_csv('/user-data/Main/logs/valid_metrics.csv',mode='a',header=False,index=False)#
        # if epoch %10 ==0:
        state = {
            
            "state_dict": Prior.state_dict()
        }

        torch.save(state, os.path.join(args.save_dir, f"model.{epoch}.pt"))

        # scheduler.step()
    # state = torch.load('/user-data/Main/save/model.5_1.pt')
    # pretrain_state_dict = state["state_dict"]
    # Prior.load_state_dict(pretrain_state_dict)
    # smil=[]
    # new=Prior.sample(n_batch=128,max_length=100)
    



    # set random seed
   

    # logger setup
    # logger = setup_logger(args)





  
    

    # valid()
