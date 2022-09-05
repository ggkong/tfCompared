import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque

from torch_jtnn import *
import rdkit

class JTVAETrainer():
    """JTVAE training application

    Args:
        train (str): Path of training data file 
        vocab (str): Path of vocablary file
        save_dir (str): Path of save directory

        load_epoch (int,optional): epoch number of trained model

        hidden_size (int,optional): Size of hidden layer.
        latent_size (int,optional): Size of latent variable.
        depthT (int,optional): depth of tree.
        depthG (int,optional): depth of graph.

        lr (float,optional): 
        clip_norm (float,optional):
        beta (float,optional):
        step_beta (float,optional):
        max_beta (float,optional):
        warmup (int,optional):

        epoch (int,optional):
        batch_size (int,optional):
        anneal_rate (float,optional):
        anneal_iter (int,optional):
        kl_anneal_iter (int,optional):
        print_iter (int,optional):
        save_iter (int,optional):

    """    
    def __init__(self,**args): 

        self.train = args['train']
        self.vocab = args['vocab']
        self.save_dir = args['save_dir']

        self.load_epoch = args.get('load_epoch',0)

        # model parameter
        self.hidden_size = int(args.get('hidden_size',450))
        self.batch_size = args.get('batch_size',32)
        self.latent_size = int(args.get('latent_size',56))
        self.depthT = args.get('depthT',20)
        self.depthG = args.get('gepthG',3)

        #training parameter
        self.lr = args.get('lr',1e-3)
        self.clip_norm = args.get('clip_norm',50.0)
        self.beta = args.get('beta',0.0)
        self.step_beta = args.get('step_beta',0.002)
        self.max_beta = args.get('max_beta',1.0)
        self.warmup = args.get('warmup',40000)

        self.epoch = args.get('epoch',20)
        self.anneal_rate = args.get('anneal_rate',0.9)
        self.anneal_iter = args.get('anneal_iter',40000)
        self.kl_anneal_iter = args.get('kl_anneal_iter',2000)
        self.print_iter = args.get('print_iter',50)
        self.save_iter = args.get('save_iter',5000)


    def __call__(self):
        vocab,weights = zip(*[(x.split(',')[0].strip('\n\r'),int(x.split(',')[1].strip('\n\r'))) for x in open(self.vocab) ])
        #print(vocab)
        #print(weights)
        weights = 1/np.asarray(weights)
        print(type(weights))
        vocab = Vocab(vocab,weightes=weights)

        model = JTNNVAE(vocab, self.hidden_size, self.latent_size, self.depthT, self.depthG).cuda()
        print (model)

        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

        if self.load_epoch > 0:
            model.load_state_dict(torch.load(self.save_dir + "/model.iter-" + str(self.load_epoch)))

        print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, self.anneal_rate)
        scheduler.step()

        param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
        grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

        total_step = self.load_epoch
        beta = self.beta
        meters = np.zeros(4)

        for epoch in range(self.epoch):
            loader = MolTreeFolder(self.train, vocab, self.batch_size, num_workers=4)
            for batch in loader:
                total_step += 1
                try:
                    model.zero_grad()
                    loss, kl_div, wacc, tacc, sacc = model(batch, beta)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                    optimizer.step()
                except Exception as e:
                    print(e)
                    continue

                meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

                if total_step % self.print_iter == 0:
                    meters /= self.print_iter
                    print ("[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model)))
                    sys.stdout.flush()
                    meters *= 0

                if total_step % self.save_iter == 0:
                    torch.save(model.state_dict(), self.save_dir + "/model.iter-" + str(total_step))

                if total_step % self.anneal_iter == 0:
                    scheduler.step()
                    print ("learning rate: %.6f" % scheduler.get_lr()[0])

                if total_step % self.kl_anneal_iter == 0 and total_step >= self.warmup:
                    beta = min(self.max_beta, beta + self.step_beta)

    @staticmethod
    def chart_parser(parser=None):
        parser = parser or argparse.ArgumentParser()
        parser.add_argument('--train', required=True)
        parser.add_argument('--vocab', required=True)
        parser.add_argument('--save_dir', required=True)
        parser.add_argument('--load_epoch', type=int, default=0)

        parser.add_argument('--hidden_size', type=int, default=450)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--latent_size', type=int, default=56)
        parser.add_argument('--depthT', type=int, default=20)
        parser.add_argument('--depthG', type=int, default=3)

        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--clip_norm', type=float, default=50.0)
        parser.add_argument('--beta', type=float, default=0.0)
        parser.add_argument('--step_beta', type=float, default=0.002)
        parser.add_argument('--max_beta', type=float, default=1.0)
        parser.add_argument('--warmup', type=int, default=40000)

        parser.add_argument('--epoch', type=int, default=20)
        parser.add_argument('--anneal_rate', type=float, default=0.9)
        parser.add_argument('--anneal_iter', type=int, default=40000)
        parser.add_argument('--kl_anneal_iter', type=int, default=2000)
        parser.add_argument('--print_iter', type=int, default=50)
        parser.add_argument('--save_iter', type=int, default=5000)
        return parser

def jtvae_trainer(train,vocab,save_dir,**kwargs):
    """JTVAE training application

    Args:
        train (str): Path of training data file 
        vocab (str): Path of vocablary file
        save_dir (str): Path of save directory

        load_epoch (int,optional): epoch number of trained model

        hidden_size (int,optional): Size of hidden layer.
        latent_size (int,optional): Size of latent variable.
        depthT (int,optional): depth of tree.
        depthG (int,optional): depth of graph.

        lr (float,optional): 
        clip_norm (float,optional):
        beta (float,optional):
        step_beta (float,optional):
        max_beta (float,optional):
        warmup (int,optional):

        epoch (int,optional):
        batch_size (int,optional):
        anneal_rate (float,optional):
        anneal_iter (int,optional):
        kl_anneal_iter (int,optional):
        print_iter (int,optional):
        save_iter (int,optional):

    """   
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    JTVAETrainer(train=train,vocab=vocab,save_dir=save_dir,**kwargs)()


def main():
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = JTVAETrainer.chart_parser()

    args = parser.parse_args()

    JTVAETrainer(**args.__dict__)()

if __name__ == "__main__":
    main()
