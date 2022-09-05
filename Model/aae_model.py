'''
Author: 成凯阳
Date: 2022-03-19 09:54:26
LastEditors: 成凯阳
LastEditTime: 2022-05-01 07:37:46
FilePath: /Main/Model/aae_model.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
import numpy
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, embedding_layer, hidden_size, num_layers,
                 bidirectional, dropout, latent_size):
        super(Encoder, self).__init__()

        self.embedding_layer = embedding_layer
        self.lstm_layer = nn.LSTM(embedding_layer.embedding_dim,
                                  hidden_size, num_layers,
                                  batch_first=True, dropout=dropout,
                                  bidirectional=bidirectional)
        self.linear_layer = nn.Linear(
            (int(bidirectional) + 1) * num_layers * hidden_size,
            latent_size
        )

    def forward(self, x, lengths):
        batch_size = x.shape[0]

        x = self.embedding_layer(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        _, (_, x) = self.lstm_layer(x)
        x = x.permute(1, 2, 0).contiguous().view(batch_size, -1)
        x = self.linear_layer(x)
        x_h=x.data.cpu().numpy()
        # numpy.save('aae.npy',x_h)

        return x


class Decoder(nn.Module):
    def __init__(self, embedding_layer, hidden_size,
                 num_layers, dropout, latent_size):
        super(Decoder, self).__init__()

        self.latent2hidden_layer = nn.Linear(latent_size, hidden_size)
        self.embedding_layer = embedding_layer
        self.lstm_layer = nn.LSTM(embedding_layer.embedding_dim,
                                  hidden_size, num_layers,
                                  batch_first=True, dropout=dropout)
        self.linear_layer = nn.Linear(hidden_size,
                                      embedding_layer.num_embeddings)

    def forward(self, x, lengths, states, is_latent_states=False):
        if is_latent_states:
            c0 = self.latent2hidden_layer(states)
            c0 = c0.unsqueeze(0).repeat(self.lstm_layer.num_layers, 1, 1)
            h0 = torch.zeros_like(c0)
            states = (h0, c0)

        x = self.embedding_layer(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        x, states = self.lstm_layer(x, states)
        x, lengths = pad_packed_sequence(x, batch_first=True)
        x = self.linear_layer(x)

        return x, lengths, states


class Discriminator(nn.Module):
    def __init__(self, input_size, layers):
        super(Discriminator, self).__init__()

        in_features = [input_size] + layers
        out_features = layers + [1]

        self.layers_seq = nn.Sequential()
        for k, (i, o) in enumerate(zip(in_features, out_features)):
            self.layers_seq.add_module('linear_{}'.format(k), nn.Linear(i, o))
            if k != len(layers):
                self.layers_seq.add_module('activation_{}'.format(k),
                                           nn.ELU(inplace=True))

    def forward(self, x):
        return self.layers_seq(x)


class AAE(nn.Module):
    def __init__(self, vocabulary, config):
        super(AAE, self).__init__()

        self.vocabulary = vocabulary
        self.latent_size = config.latent_size

        self.embeddings = nn.Embedding(len(vocabulary),
                                       config.embedding_size,
                                       padding_idx=vocabulary.vocab['<pad>'])
        self.encoder = Encoder(self.embeddings, config.encoder_hidden_size,
                               config.num_layers,
                               config.encoder_bidirectional,
                               config.dropout,
                               config.latent_size)
        self.decoder = Decoder(self.embeddings,
                               config.decoder_hidden_size,
                               config.decoder_num_layers,
                               config.dropout,
                               config.latent_size)
        self.discriminator = Discriminator(config.latent_size,
                                           config.discriminator_layers)

    # @property
    # def device(self):
    #     return next(self.parameters()).device
    # def forward(self, x, lengths):
    #     x=self.encoder(x,lengths)
    #     x=self.decoder(x, lengths, states, is_latent_states=False)


    def encoder_forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decoder_forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def discriminator_forward(self, *args, **kwargs):
        return self.discriminator(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def string2tensor(self, string, device='model'):
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(
            ids, dtype=torch.long,
            device=self.device if device == 'model' else device
        )

        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

        return string

  

    def sample(self, n_batch, args,max_len=100):
        with torch.no_grad():
            samples = []
            lengths = torch.zeros(
                n_batch, dtype=torch.long, device=args.cuda
            )

            # states = self.sample_latent(n_batch)
            states=torch.randn(n_batch, args.latent_size, device=args.cuda)
            prevs = torch.empty(
                n_batch, 1, dtype=torch.long, device=args.cuda
            ).fill_(self.vocabulary.vocab['<bos>'])
            one_lens = torch.ones(n_batch, dtype=torch.long,
                                  device=args.cuda)
            is_end = torch.zeros(n_batch, dtype=torch.uint8,
                                 device=args.cuda)

            for i in range(max_len):
                logits, _, states = self.decoder(prevs, one_lens,
                                                 states, i == 0)
                logits = torch.softmax(logits, 2)
                shape = logits.shape[:-1]
                logits = logits.contiguous().view(-1, logits.shape[-1])
                currents = torch.distributions.Categorical(logits).sample()
                currents = currents.view(shape)

                is_end[currents.view(-1) == self.vocabulary.vocab['<eos>']] = 1
                if is_end.sum() == max_len:
                    break

                currents[is_end, :] = self.vocabulary.vocab['<pad>']
                samples.append(currents.cpu())
                lengths[~is_end] += 1

                prevs = currents
            

            # if len(samples):
            #     samples = torch.cat(samples, dim=-1)
            #     samples = [
            #         self.tensor2string(t[:l])
            #         for t, l in zip(samples, lengths)
            #     ]
            # else:
            #     samples = ['' for _ in range(n_batch)]

            return torch.cat(samples, dim=-1)
class Prop(nn.Module):
    def __init__(self, model:AAE, hidden_size):
        super(Prop, self).__init__()
        self.model = model
        # self.latent_size = model.b_size + model.a_size
        self.hidden_size = hidden_size

        # vh = (self.latent_size,) + tuple(hidden_size) + (1,)
        vh =tuple(hidden_size) + (1,)
        modules = []
        for i in range(len(vh)-1):
            modules.append(nn.Linear(vh[i], vh[i+1]))
            if i < len(vh) - 2:
                modules.append(nn.Tanh())
                # modules.append(nn.ReLU())
        self.propNN = nn.Sequential(*modules)

    def encode(self, adj, x):
        with torch.no_grad():
            self.model.eval()
            adj_normalized = rescale_adj(adj).to(adj)
            z, sum_log_det_jacs = self.model(adj, x, adj_normalized)  # z = [h, adj_h]
            h = torch.cat([z[0].reshape(z[0].shape[0], -1), z[1].reshape(z[1].shape[0], -1)], dim=1)
        return h, sum_log_det_jacs

 

    def forward(self, x, lengths):
        h = self.encoder(x,lengths)
        output = self.propNN(h)  # do I need to add nll of the unsupervised part? or just keep few epoch? see the results
        return output
