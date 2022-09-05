'''
Author: 成凯阳
Date: 2022-03-27 02:14:01
LastEditors: 成凯阳
LastEditTime: 2022-03-30 01:48:39
FilePath: /Main/Model/organ_model.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
# from Model.model import CharRNN

from Utils.utils.metric_reward import MetricsReward
import torch.nn.utils.rnn as rnn_utils
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


class Discriminator(nn.Module):
    def __init__(self,embedding_layer,  config,convs, dropout=0):
        super(Discriminator, self).__init__()

        self.embedding_layer = embedding_layer
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(1, f, kernel_size=(
                n, self.embedding_layer.embedding_dim)
                       ) for f, n in convs])
        sum_filters = sum([f for f, _ in convs])
        self.highway_layer = nn.Linear(sum_filters, sum_filters)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(sum_filters, 1)
        self.device=config.cuda

    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.unsqueeze(1)

        convs = [F.elu(conv_layer(x)).squeeze(3)
                 for conv_layer in self.conv_layers]
        x = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in convs]
        x = torch.cat(x, dim=1)

        h = self.highway_layer(x)
        t = torch.sigmoid(h)
        x = t * F.elu(h) + (1 - t) * x
        x = self.dropout_layer(x)
        out = self.output_layer(x)

        return out
def reback(model,toens,n_batch):
    smis=[]
   

    for i in range(n_batch):
        stri=''
        for j in range(len(toens[i].cuda().cpu().numpy())):
            smi=model.vocabulary.reversed_vocab[int(toens[i][j])]
            stri+=smi
        smis.append(stri)
       

        

    return smis 

class ORGAN(nn.Module):
    def __init__(self, vocabulary, config):
        super(ORGAN, self).__init__()

        self.metrics_reward = MetricsReward(
            config.n_ref_subsample, config.rollouts,
            config.n_jobs, config.additional_rewards)
        self.reward_weight = config.reward_weight

        self.vocabulary = vocabulary
        self.device=config.cuda

        self.generator_embeddings = nn.Embedding(
            len(vocabulary), config.embedding_size, padding_idx=vocabulary.vocab['<pad>']
        )
        self.discriminator_embeddings = nn.Embedding(
            len(vocabulary), config.embedding_size, padding_idx=vocabulary.vocab['<pad>']
        )
        # self.generator = Generator(
        #     self.generator_embeddings, config.hidden,
        #     config.num_layers, config.dropout,config.cuda
        # )
        self.generator=CharRNN(vocabulary, config)
        self.discriminator = Discriminator(
            self.discriminator_embeddings,
           config, config.discriminator_layers, config.discriminator_dropout
        )

    # @property
    # def device(self):
    #     return next(self.parameters()).device

    def generator_forward(self, *args, **kwargs):
        return self.generator(*args, **kwargs)

    def discriminator_forward(self, *args, **kwargs):
        return self.discriminator(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def string2tensor(self, string, device='model'):
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(ids, dtype=torch.long,
                              device=self.device
                              if device == 'model' else device)

        return tensor

    def tensor2string(self, tensor):
        str=''
        ids = tensor.tolist()
        for i in range(len(ids)):
            str+=self.vocabulary.reversed_vocab[ids[i]]
        # string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

        return str[5:-5]

    def _proceed_sequences(self, prevs, states, max_len):
        with torch.no_grad():
            n_sequences = prevs.shape[0]

            sequences = []
            lengths = torch.zeros(n_sequences,
                                  dtype=torch.long,
                                  device=prevs.device)

            one_lens = torch.ones(n_sequences,
                                  dtype=torch.long,
                                  device=prevs.device)
            is_end = prevs.eq(self.vocabulary.vocab['<eos>']).view(-1)

            for _ in range(max_len):
                outputs, _, states = self.generator(prevs, one_lens, states)
                probs = F.softmax(outputs, dim=-1).view(n_sequences, -1)
                currents = torch.multinomial(probs, 1)

                currents[is_end, :] = self.vocabulary.vocab['<pad>']
                sequences.append(currents)
                lengths[~is_end] += 1

                is_end[currents.view(-1) ==self.vocabulary.vocab['<eos>']] = 1
                if is_end.sum() == n_sequences:
                    break

                prevs = currents

            sequences = torch.cat(sequences, dim=-1)

        return sequences, lengths

    def rollout(self,
                n_samples, n_rollouts, ref_smiles,
                ref_mols, max_len=100):
        with torch.no_grad():
            sequences = []
            rewards = []
            lengths = torch.zeros(n_samples,
                                  dtype=torch.long,
                                  device=self.device)

            one_lens = torch.ones(n_samples,
                                  dtype=torch.long,
                                  device=self.device)
            prevs = torch.empty(n_samples, 1,
                                dtype=torch.long,
                                device=self.device).fill_(self.vocabulary.vocab['<bos>'])
            is_end = torch.zeros(n_samples,
                                 dtype=torch.uint8,
                                 device=self.device)
            states = None

            sequences.append(prevs)
            lengths += 1

            for current_len in range(max_len):
                outputs, _, states = self.generator(prevs, one_lens, states)
                probs = F.softmax(outputs, dim=-1).view(n_samples, -1)
                currents = torch.multinomial(probs, 1)

                currents[is_end, :] = self.vocabulary.vocab['<pad>']
                sequences.append(currents)
                lengths[~is_end] += 1

                rollout_prevs = currents[~is_end, :].repeat(n_rollouts, 1)
                rollout_states = (
                    states[0][:, ~is_end, :].repeat(1, n_rollouts, 1),
                    states[1][:, ~is_end, :].repeat(1, n_rollouts, 1)
                )
                rollout_sequences, rollout_lengths = self._proceed_sequences(
                    rollout_prevs, rollout_states, max_len - current_len
                )

                rollout_sequences = torch.cat(
                    [s[~is_end, :].repeat(n_rollouts, 1)
                     for s in sequences] + [rollout_sequences],
                    dim=-1
                )
                rollout_lengths += lengths[~is_end].repeat(n_rollouts)

                rollout_rewards = torch.sigmoid(
                    self.discriminator(rollout_sequences).detach()
                )

                if self.metrics_reward is not None and self.reward_weight > 0:
                    strings = [
                        self.tensor2string(t[:l])
                        for t, l in zip(rollout_sequences, rollout_lengths)
                    ]
                    obj_rewards = torch.tensor(
                        self.metrics_reward(strings, ref_smiles, ref_mols),
                        device=rollout_rewards.device
                    ).view(-1, 1)
                    rollout_rewards = (
                        rollout_rewards * (1 - self.reward_weight) +
                        obj_rewards * self.reward_weight
                    )

                current_rewards = torch.zeros(n_samples, device=self.device,dtype=float)
                rollout_rewards=rollout_rewards.double()

                current_rewards[~is_end] = rollout_rewards.view(
                    n_rollouts, -1
                ).mean(dim=0)
                rewards.append(current_rewards.view(-1, 1))

                is_end[currents.view(-1) == self.vocabulary.vocab['<eos>']] = 1
                if is_end.sum() == n_samples:
                    break

                prevs = currents

            sequences = torch.cat(sequences, dim=1)
            rewards = torch.cat(rewards, dim=1)

        return sequences, rewards, lengths

    def sample_tensor(self, n, max_len=100):
        prevs = torch.empty(n, 1,
                            dtype=torch.long,
                            device=self.device).fill_(self.vocabulary.vocab['<bos>'])
        samples, lengths = self._proceed_sequences(prevs, None, max_len)

        samples = torch.cat([prevs, samples], dim=-1)
        lengths += 1

        return samples, lengths

    def sample(self, batch_n, args,max_len=100):
        samples, lengths = self.sample_tensor(batch_n, max_len)
        samples = [t[:l] for t, l in zip(samples, lengths)]

        return samples