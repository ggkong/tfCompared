'''
Author: 成凯阳
Date: 2022-04-10 09:19:36
LastEditors: 成凯阳
LastEditTime: 2022-04-24 03:06:58
FilePath: /Main/Model/graphvae_model.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

# from Script.Vae_Script.train_vae import Prior
logabs = lambda x: torch.log(torch.abs(x))
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, u, activation, edge_type_num, dropout_rate=0.):
        super(GraphConvolutionLayer, self).__init__()
        self.edge_type_num = edge_type_num
        self.u = u
        self.adj_list = nn.ModuleList()
        for _ in range(self.edge_type_num):
            self.adj_list.append(nn.Linear(in_features, u))
        self.linear_2 = nn.Linear(in_features, u)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, n_tensor, adj_tensor, h_tensor=None):
        if h_tensor is not None:
            annotations = torch.cat((n_tensor, h_tensor), -1)
        else:
            annotations = n_tensor

        output = torch.stack([self.adj_list[i](annotations) for i in range(self.edge_type_num)], 1)
        output = torch.matmul(adj_tensor, output)
        out_sum = torch.sum(output, 1)
        out_linear_2 = self.linear_2(annotations)
        output = out_sum + out_linear_2
        output = self.activation(output) if self.activation is not None else output
        output = self.dropout(output)
        return output


class MultiGraphConvolutionLayers(nn.Module):
    def __init__(self, in_features, units, activation, edge_type_num, with_features=False, f=0, dropout_rate=0.):
        super(MultiGraphConvolutionLayers, self).__init__()
        self.conv_nets = nn.ModuleList()
        self.units = units
        in_units = []
        if with_features:
            for i in range(len(self.units)):
                in_units = list([x + in_features for x in self.units])
            for u0, u1 in zip([in_features+f] + in_units[:-1], self.units):
                self.conv_nets.append(GraphConvolutionLayer(u0, u1, activation, edge_type_num, dropout_rate))
        else:
            for i in range(len(self.units)):
                in_units = list([x + in_features for x in self.units])
            for u0, u1 in zip([in_features] + in_units[:-1], self.units):
                self.conv_nets.append(GraphConvolutionLayer(u0, u1, activation, edge_type_num, dropout_rate))

    def forward(self, n_tensor, adj_tensor, h_tensor=None):
        hidden_tensor = h_tensor
        for conv_idx in range(len(self.units)):
            hidden_tensor = self.conv_nets[conv_idx](n_tensor, adj_tensor, hidden_tensor)
        return hidden_tensor


class GraphConvolution(nn.Module):
    def __init__(self, in_features, graph_conv_units, edge_type_num, with_features=False, f_dim=0, dropout_rate=0.):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.graph_conv_units = graph_conv_units
        self.activation_f = torch.nn.Tanh()
        self.multi_graph_convolution_layers = \
            MultiGraphConvolutionLayers(in_features, self.graph_conv_units, self.activation_f, edge_type_num,
                                        with_features, f_dim, dropout_rate)

    def forward(self, n_tensor, adj_tensor, h_tensor=None):
        output = self.multi_graph_convolution_layers(n_tensor, adj_tensor, h_tensor)
        return output



class GraphAggregation(nn.Module):
    def __init__(self, in_features, aux_units, activation, with_features=False, f_dim=0,
                 dropout_rate=0.):
        super(GraphAggregation, self).__init__()
        self.with_features = with_features
        self.activation = activation
        if self.with_features:
            self.i = nn.Sequential(nn.Linear(in_features+f_dim, aux_units),
                                   nn.Sigmoid())
            j_layers = [nn.Linear(in_features+f_dim, aux_units)]
            if self.activation is not None:
                j_layers.append(self.activation)
            self.j = nn.Sequential(*j_layers)
        else:
            self.i = nn.Sequential(nn.Linear(in_features, aux_units),
                                   nn.Sigmoid())
            j_layers = [nn.Linear(in_features, aux_units)]
            if self.activation is not None:
                j_layers.append(self.activation)
            self.j = nn.Sequential(*j_layers)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, n_tensor, out_tensor, h_tensor=None):
        if h_tensor is not None:
            annotations = torch.cat((out_tensor, h_tensor, n_tensor), -1)
        else:
            annotations = torch.cat((out_tensor, n_tensor), -1)
        # The i here seems to be an attention.
        i = self.i(annotations)
        j = self.j(annotations)
        output = torch.sum(torch.mul(i, j), 1)
        if self.activation is not None:
            output = self.activation(output)
        output = self.dropout(output)

        return output

    
class MultiDenseLayer(nn.Module):
    def __init__(self, in_features, linear_units, activation=None, dropout_rate=0.):
        super(MultiDenseLayer, self).__init__()
        layers = []
        for c0, c1 in zip([in_features] + linear_units[:-1], linear_units):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Dropout(dropout_rate))
            if activation is not None:
                layers.append(activation)
        self.linear_layer = nn.Sequential(*layers)

    def forward(self, inputs):
        h = self.linear_layer(inputs)
        return h
class graphEncoder(nn.Module):
    def __init__(self, conv_dim, m_dim, b_dim, linear_dim, vertexes, edges, nodes, with_features=False, f_dim=0, dropout_rate=0.):
        super(graphEncoder, self).__init__()
        # gcn时graph channel 为bond type， 生成新graph时channel为bond+1
        # b_dim = edges - 1
        graph_conv_dim, aux_dim = conv_dim
        self.activation_f = torch.nn.Tanh()
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dim, b_dim, with_features, f_dim, dropout_rate)
        self.agg_layer = GraphAggregation(graph_conv_dim[-1] + m_dim, aux_dim, self.activation_f)
        self.norm1 = nn.BatchNorm1d(aux_dim) ###
        self.multi_dense_layer = MultiDenseLayer(aux_dim, linear_dim, torch.nn.Tanh())
#         self.norm2 = nn.BatchNorm1d(linear_dim[-1]) ###

        self.emb_mean = nn.Linear(linear_dim[-1], edges*vertexes*vertexes+vertexes*nodes)
        self.emb_logvar = nn.Linear(linear_dim[-1], edges*vertexes*vertexes+vertexes*nodes)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, adj, node):
        adj = adj[:, :-1, :, :]
        output = self.gcn_layer(node, adj)
        output = self.agg_layer(output, node)
#         output = self.norm1(output)
        output = self.multi_dense_layer(output)
        # h_mu = self.emb_mean(output)
        # h_logvar = self.emb_logvar(output)
        # h = self.reparameterize(h_mu, h_logvar)      
        return output
def rescale_adj(adj, type='all'):
    # Previous paper didn't use rescale_adj.
    # In their implementation, the normalization sum is: num_neighbors = F.sum(adj, axis=(1, 2))
    # In this implementation, the normaliztion term is different
    # raise NotImplementedError
    # (256,4,9, 9):
    # 4: single, double, triple, and bond between disconnected atoms (negative mask of sum of previous)
    # 1-adj[i,:3,:,:].sum(dim=0) == adj[i,4,:,:]
    # usually first 3 matrices have no diagnal, the last has.
    # A_prime = self.A + sp.eye(self.A.shape[0])
    if type == 'view':
        out_degree = adj.sum(dim=-1)
        out_degree_sqrt_inv = out_degree.pow(-1)
        out_degree_sqrt_inv[out_degree_sqrt_inv == float('inf')] = 0
        adj_prime = out_degree_sqrt_inv.unsqueeze(-1) * adj  # (256,4,9,1) * (256, 4, 9, 9) = (256, 4, 9, 9)
    else:  # default type all
        num_neighbors = adj.sum(dim=(1, 2)).float()
        num_neighbors_inv = num_neighbors.pow(-1)
        num_neighbors_inv[num_neighbors_inv == float('inf')] = 0
        adj_prime = num_neighbors_inv[:, None, None, :] * adj
    return adj_prime
class GVAE(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()

        self.vocabulary = vocab
        self.graphenc = graphEncoder(config.enc_conv_dim, config.a_n_type, config.b_n_type-1, config.enc_linear_dim, config.a_n_node, config.b_n_type, config.a_n_type)
        # Special symbols
        # for ss in ('bos', 'eos', 'unk', 'pad'):
        #     setattr(self, ss, getattr(vocab, ss))

        # Word embeddings layer
        n_vocab, d_emb = len(vocab), vocab.vectors.size(1)
        self.x_emb = nn.Embedding(n_vocab, d_emb, vocab.vocab['<pad>'])
        self.x_emb.weight.data.copy_(vocab.vectors)
        if config.freeze_embeddings:
            self.x_emb.weight.requires_grad = False

        # Encoder
        if config.q_cell == 'gru':
            self.encoder_rnn = nn.GRU(
                d_emb,
                config.q_d_h,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0,
                bidirectional=config.q_bidir
            )
        else:
            raise ValueError(
                "Invalid q_cell type, should be one of the ('gru',)"
            )

        q_d_last = config.q_d_h * (2 if config.q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last+config.enc_linear_dim[0], config.d_z)
        self.q_logvar = nn.Linear(q_d_last+config.enc_linear_dim[0], config.d_z)

        # Decoder
        if config.d_cell == 'gru':
            self.decoder_rnn = nn.GRU(
                d_emb + config.d_z,
                config.d_d_h,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0
            )
        else:
            raise ValueError(
                "Invalid d_cell type, should be one of the ('gru',)"
            )
        self.device=config.cuda

        self.decoder_lat = nn.Linear(config.d_z, config.d_d_h)
        self.decoder_fc = nn.Linear(config.d_d_h, n_vocab)

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
        ])
        self.decoder = nn.ModuleList([
            self.decoder_rnn,
            self.decoder_lat,
            self.decoder_fc
        ])
        self.vae = nn.ModuleList([
            self.x_emb,
            self.encoder,
            self.decoder
        ])



    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

        return string

    def forward(self, x,bond,node,adj):
        """Do the VAE forward step

        :param x: list of tensors of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder(x,bond,node,adj)

        # Decoder: x, z -> recon_loss
        recon_loss = self.forward_decoder(x, z)

        return kl_loss, recon_loss

    def forward_encoder(self, x,bond,node,adj):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """
        hidden1 = self.graphenc(bond,node)
        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)

        _, h = self.encoder_rnn(x, None)

        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)
        h = torch.cat([h, hidden1], dim=1).float()

        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps
        z_h=z.detach().cpu().numpy()
        # numpy.save('vae.npy',z_h)

        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()

        return z, kl_loss

    def forward_decoder(self, x, z):
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        """

        lengths = [len(i_x) for i_x in x]

        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.vocabulary.vocab['<pad>'])
        x_emb = self.x_emb(x)

        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True)

        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

        output, _ = self.decoder_rnn(x_input, h_0)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)
        # lie=[]
        
        # for i in y:
            
        #     smi=''
        #     smitarget=''
        #     for j in i:

        #         y_h = F.softmax(j , dim=-1)

        #         w = torch.multinomial(y_h, 1)
        #         x[:, 1:]
                
        #         token=self.vocabulary.reversed_vocab[int(w)]
        #         smi+=token
        #     smi=smi.split('<eos>')[0]
        #     lie.append(smi)
            


        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.vocabulary.vocab['<pad>']
        )

        return recon_loss

    def sample_z_prior(self,args, n_batch):
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return torch.randn(n_batch, self.q_mu.out_features,
                           device=args.cuda)

    def sample(self, n_batch,args, max_len=100, z=None, temp=1.0):
        """Generating n_batch samples in eval mode (`z` could be
        not on same device)

        :param n_batch: number of sentences to generate
        :param max_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param temp: temperature of softmax
        :return: list of tensors of strings, samples sequence x
        """
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(args,n_batch)
            z = z.to(args.cuda)
            z_0 = z.unsqueeze(1)

            # Initial values
            h = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.vocabulary.vocab['<bos>'], device=args.cuda).repeat(n_batch)
            x = torch.tensor([self.vocabulary.vocab['<pad>']], device=args.cuda).repeat(n_batch,
                                                                    max_len)
            x[:, 0] = self.vocabulary.vocab['<bos>']
            end_pads = torch.tensor([max_len], device=args.cuda).repeat(
                n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.uint8,
                                   device=args.cuda)

            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder_rnn(x_input, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)

                w = torch.multinomial(y, 1)[:, 0]
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.vocabulary.vocab['<eos>'])
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
         
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, 1:end_pads[i]-1])
            return new_x