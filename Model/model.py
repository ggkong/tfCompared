'''
Author: 成凯阳
Date: 2022-03-11 14:45:13
LastEditors: 成凯阳
LastEditTime: 2022-06-20 08:45:46
FilePath: /Main/Model/model.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
import numpy as np


from Utils.utils.metric import NLLLosss

class CharRNN(nn.Module):

    def __init__(self, vocabulary, config):
        super(CharRNN, self).__init__()

        self.vocabulary = vocabulary
        self.hidden_size = config.hidden
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.device=config.cuda
        self.vocab_size = self.input_size = self.output_size = len(vocabulary)

        self.embedding_layer = nn.Embedding(self.vocab_size, self.vocab_size,
                                            padding_idx=vocabulary.vocab['<pad>'])
        self.lstm_layer = nn.LSTM(self.input_size, self.hidden_size,
                                  self.num_layers, dropout=self.dropout,
                                  batch_first=True)
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)

    # @property
    # def device(self):
    #     return next(self.parameters()).device

    def forward(self, x, lengths, hiddens=None):
        x = self.embedding_layer(x)
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True)
        x, hiddens = self.lstm_layer(x, hiddens)
        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)
        x = self.linear_layer(x)

        return x, lengths, hiddens
   



  

    def sample(self, n_batch, max_length=70):
        with torch.no_grad():
            # actions = torch.LongTensor(n_batch, max_length).to(self.device)
            starts = [torch.tensor([self.vocabulary.vocab['<bos>']],
                                   dtype=torch.long,
                                   device=self.device)
                      for _ in range(n_batch)]

            starts = torch.tensor(starts, dtype=torch.long,
                                  device=self.device).unsqueeze(1)

            new_smiles_list = [
                torch.tensor(self.vocabulary.vocab['<pad>'], dtype=torch.long,
                             device=self.device).repeat(max_length + 2)
                for _ in range(n_batch)]

            for i in range(n_batch):
                new_smiles_list[i][0] = self.vocabulary.vocab['<bos>']

            len_smiles_list = [1 for _ in range(n_batch)]
            lens = torch.tensor([1 for _ in range(n_batch)],
                                dtype=torch.long, device=self.device)
            end_smiles_list = [False for _ in range(n_batch)]

            hiddens = None
            zong=[]
            zong.append(starts)
            log_probs = torch.tensor([0 for _ in range(n_batch)],
                                dtype=torch.float, device=self.device)
            entropy = torch.tensor([0 for _ in range(n_batch)],
                                dtype=torch.float, device=self.device)
       
            
            for i in range(1, max_length + 1):
                output, _, hiddens = self.forward(starts, lens, hiddens)
                prob = F.softmax(output.view(-1, output.shape[-1]))
                
                # log_prob = F.log_softmax(output)
                log_prob=F.log_softmax(output.view(-1, output.shape[-1]))
                # log_probs +=  NLLLosss(log_prob, starts.view(-1))
                # entropy += -torch.sum((log_prob * prob), 1)
                x = torch.multinomial(prob,1).view(-1)
                # log_probs +=  NLLLosss(log_prob, x)

                # probabilities
                probs = [F.softmax(o, dim=-1) for o in output]

                # sample from probabilities
                ind_tops = [torch.multinomial(p, 1) for p in probs]
                # log_probs +=  NLLLosss(log_prob, starts.view(-1))
                entropy += -torch.sum((log_prob * prob), 1)
                imx=torch.cat(ind_tops,0)
                zong.append(imx)
            

                for j, top in enumerate(ind_tops):
                    if not end_smiles_list[j]:
                        top_elem = top[0].item()
                        if top_elem == self.vocabulary.vocab['<eos>']:
                            end_smiles_list[j] = True

                        new_smiles_list[j][i] = top_elem
                        len_smiles_list[j] = len_smiles_list[j] + 1

                starts = torch.tensor(ind_tops, dtype=torch.long,
                                      device=self.device).unsqueeze(1)
            zong=torch.cat(zong,1)
            new=[]
            token=[]
          
            for i in range(n_batch):
                flag=False
                count=0
                eosflag=False
                wordz=[]
        
                if torch.where(zong[i]==self.vocabulary.vocab['<eos>'])[0].shape==torch.Size([0]):

                    eosflag=True
                    new.append(zong[i][1:])
                    xiangliang=zong[i][1:]
                    word=[]
                    for i in range(len(xiangliang)):
                        word.append(self.vocabulary.reversed_vocab[int(xiangliang[i])])
                    wordz.append(word)
                    # cha=
                    # print(i)
                    continue
           
                if not eosflag and not flag:
                    j= torch.where(zong[i]==self.vocabulary.vocab['<eos>'])[0][0]
                    new.append(zong[i][1:j])
                    xiangliang=zong[i][1:j]
                    word=[]
                    for i in range(len(xiangliang)):
                        word.append(self.vocabulary.reversed_vocab[int(xiangliang[i])])
                    wordz.append(word)

                    flag=True
                      
                        


                   
                 

                   
                

                 



            new_smiles_list = [new_smiles_list[i][:l]
                               for i, l in enumerate(len_smiles_list)]
        
            return new
    
    def likehood(self,seq,n_batch):
     
            # actions = torch.LongTensor(n_batch, max_length).to(self.device)
            starts = [torch.tensor([self.vocabulary.vocab['<bos>']],
                                   dtype=torch.long,
                                   device=self.device)
                      for _ in range(n_batch)]

            starts = torch.tensor(starts, dtype=torch.long,
                                  device=self.device).unsqueeze(1)

            # new_smiles_list = [
            #     torch.tensor(self.vocabulary.vocab['<pad>'], dtype=torch.long,
            #                  device=self.device).repeat(max_length + 2)
            #     for _ in range(n_batch)]

            # for i in range(n_batch):
            #     new_smiles_list[i][0] = self.vocabulary.vocab['<bos>']

            len_smiles_list = [1 for _ in range(n_batch)]
            lens = torch.tensor([1 for _ in range(n_batch)],
                                dtype=torch.long, device=self.device)
            end_smiles_list = [False for _ in range(n_batch)]

            hiddens = None
            zong=[]
            zong.append(starts)
            log_probs = torch.tensor([0 for _ in range(n_batch)],
                                dtype=torch.float, device=self.device)
            entropy = torch.tensor([0 for _ in range(n_batch)],
                                dtype=torch.float, device=self.device)
            max_length=len(seq[0])
            batch_size=len(seq)
            start_token = torch.tensor([1 for _ in range(n_batch)],
                                dtype=torch.long, device=self.device)
            
            start_token[:] = self.vocabulary.vocab['<bos>']
            start_token=torch.unsqueeze(start_token,1)
         
            start_token = torch.cat((start_token, seq[:, :-1]), 1)
            
       
            
            for i in range(max_length):
                # start=torch.unsqueeze(start_token[:,i],1)
                output, _, hiddens = self.forward(torch.unsqueeze(start_token[:,i],1), lens, hiddens)
                prob = F.softmax(output.view(-1, output.shape[-1]))
                
                # log_prob = F.log_softmax(output)
                log_prob=F.log_softmax(output.view(-1, output.shape[-1]))
                log_probs +=  NLLLosss(prob, seq[:,i])
                entropy += -torch.sum((log_prob * prob), 1)
                # start_token=seq[:,i]
                # start=torch.unsqueeze(start_token,1)

                # probabilities
              
            return log_probs,entropy
    # def samples(self, n_batch, max_length=100):
    
    #         # actions = torch.LongTensor(n_batch, max_length).to(self.device)
    #         starts = [torch.tensor([self.vocabulary.vocab['<bos>']],
    #                                dtype=torch.long,
    #                                device=self.device)
    #                   for _ in range(n_batch)]

    #         starts = torch.tensor(starts, dtype=torch.long,
    #                               device=self.device).unsqueeze(1)

    #         new_smiles_list = [
    #             torch.tensor(self.vocabulary.vocab['<pad>'], dtype=torch.long,
    #                          device=self.device).repeat(max_length + 2)
    #             for _ in range(n_batch)]

    #         for i in range(n_batch):
    #             new_smiles_list[i][0] = self.vocabulary.vocab['<bos>']

    #         len_smiles_list = [1 for _ in range(n_batch)]
    #         lens = torch.tensor([1 for _ in range(n_batch)],
    #                             dtype=torch.long, device=self.device)
    #         end_smiles_list = [False for _ in range(n_batch)]

    #         hiddens = None
    #         zong=[]
    #         zong.append(starts)
    #         log_probs = torch.tensor([0 for _ in range(n_batch)],
    #                             dtype=torch.float, device=self.device)
    #         entropy = torch.tensor([0 for _ in range(n_batch)],
    #                             dtype=torch.float, device=self.device)
       
            
    #         for i in range(1, max_length + 1):
    #             output, _, hiddens = self.forward(starts, lens, hiddens)
    #             prob = F.softmax(output.view(-1, output.shape[-1]))
                
    #             # log_prob = F.log_softmax(output)
    #             log_prob=F.log_softmax(output.view(-1, output.shape[-1]))
    #             log_probs +=  NLLLosss(log_prob, starts.view(-1))
    #             entropy += -torch.sum((log_prob * prob), 1)

    #             # probabilities
    #             probs = [F.softmax(o, dim=-1) for o in output]

    #             # sample from probabilities
    #             ind_tops = [torch.multinomial(p, 1) for p in probs]
    #             imx=torch.cat(ind_tops,0)
    #             zong.append(imx)
            

    #             for j, top in enumerate(ind_tops):
    #                 if not end_smiles_list[j]:
    #                     top_elem = top[0].item()
    #                     if top_elem == self.vocabulary.vocab['<eos>']:
    #                         end_smiles_list[j] = True

    #                     new_smiles_list[j][i] = top_elem
    #                     len_smiles_list[j] = len_smiles_list[j] + 1

    #             starts = torch.tensor(ind_tops, dtype=torch.long,
    #                                   device=self.device).unsqueeze(1)
    #         zong=torch.cat(zong,1)
    #         new=[]
    #         token=[]
          
    #         for i in range(n_batch):
    #             flag=False
    #             count=0
    #             eosflag=False
    #             wordz=[]
        
    #             if torch.where(zong[i]==self.vocabulary.vocab['<eos>'])[0].shape==torch.Size([0]):

    #                 eosflag=True
    #                 new.append(zong[i][1:])
    #                 xiangliang=zong[i][1:]
    #                 word=[]
    #                 for i in range(len(xiangliang)):
    #                     word.append(self.vocabulary.reversed_vocab[int(xiangliang[i])])
    #                 wordz.append(word)
    #                 # cha=
    #                 print(i)
    #                 continue
           
    #             if not eosflag and not flag:
    #                 j= torch.where(zong[i]==self.vocabulary.vocab['<eos>'])[0][0]
    #                 new.append(zong[i][1:j])
    #                 xiangliang=zong[i][1:j]
    #                 word=[]
    #                 for i in range(len(xiangliang)):
    #                     word.append(self.vocabulary.reversed_vocab[int(xiangliang[i])])
    #                 wordz.append(word)

    #                 flag=True
                      
                        


                   
                 

                   
                

                 



    #         new_smiles_list = [new_smiles_list[i][:l]
    #                            for i, l in enumerate(len_smiles_list)]
        
    #         return new,log_probs,entropy

