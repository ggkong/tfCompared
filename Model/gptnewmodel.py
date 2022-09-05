'''
Author: 成凯阳
Date: 2022-03-11 14:45:13
LastEditors: 成凯阳
LastEditTime: 2022-05-08 04:29:54
FilePath: /Main/Model/model.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
import torch,math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
import numpy as np


from Utils.utils.metric import NLLLosss
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config,block_size):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config,block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config,block_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )
        self.lstm_layer = nn.LSTM(self.input_size, self.hidden_size,
                                  self.num_layers, dropout=self.dropout,
                                  batch_first=True)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
class GPT(nn.Module):

    def __init__(self, vocabulary, config):
        super(GPT, self).__init__()

        self.vocabulary = vocabulary
        self.hidden_size = config.hidden
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.device=config.cuda
        self.block_size=config.block_size
        self.drop=nn.Dropout(config.dropout)
        self.vocab_size = self.input_size = self.output_size = len(vocabulary)
        # self.blocks = nn.Sequential(*[Block(config,config.block_size) for _ in range(config.n_layer)])
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.vocab_size))
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
        b, t = x.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        x = self.embedding_layer(x)
        token_embeddings=x
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
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
    
 
