'''
Author: 成凯阳
Date: 2022-04-06 14:21:38
LastEditors: 成凯阳
LastEditTime: 2022-04-06 14:47:47
FilePath: /Main/Model/glow.py

Copyright (c) 2022 by 用户/公司名, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import math
logabs = lambda x: torch.log(torch.abs(x))
# from mflow.models.basic import ZeroConv2d, ActNorm, InvConv2dLU, InvConv2d, InvRotationLU, InvRotation, ActNorm2D
class ActNorm(nn.Module):
    def __init__(self, in_channel, use_shift=False, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet
        self.use_shift = use_shift

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            
            if not self.use_shift:
                self.loc.data.copy_(-mean)
                
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class ActNorm2D(nn.Module):
    def __init__(self, in_dim, use_shift=False, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_dim, 1))
        self.scale = nn.Parameter(torch.ones(1, in_dim, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet
        self.use_shift = use_shift

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )
            
            if not self.use_shift:
                self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
class AffineCoupling(nn.Module):  # delete
    def __init__(self, in_channel, hidden_channels, affine=True, mask_swap=False):  # filter_size=512,  --> hidden_channels =(512, 512)
        super(AffineCoupling, self).__init__()

        self.affine = affine
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.mask_swap=mask_swap
        # self.norms_in = nn.ModuleList()
        last_h = math.ceil(in_channel / 2)
        if affine:
            vh = tuple(hidden_channels) + (in_channel,)
        else:
#             vh = tuple(hidden_channels) + (in_channel // 2,)
            vh = tuple(hidden_channels) + (math.ceil(in_channel / 2),)

        for h in vh:
            self.layers.append(nn.Conv2d(last_h, h, kernel_size=3, padding=1))
            self.norms.append(nn.BatchNorm2d(h))  # , momentum=0.9 may change norm later, where to use norm? for the residual? or the sum
            # self.norms.append(ActNorm(in_channel=h, logdet=False)) # similar but not good
            last_h = h

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)  # (2,12,32,32) --> (2,6,32,32), (2,6,32,32)

        if self.mask_swap:
            in_a, in_b = in_b, in_a

        if self.affine:
            # log_s, t = self.net(in_a).chunk(2, 1)  # (2,12,32,32) --> (2,6,32,32), (2,6,32,32)
            s, t = self._s_t_function(in_a)
            out_b = (in_b + t) * s   #  different affine bias , no difference to the log-det # (2,6,32,32) More stable, less error
            # out_b = in_b * s + t
            logdet = torch.sum(torch.log(torch.abs(s)).view(input.shape[0], -1), 1)
        else:  # add coupling
            # net_out = self.net(in_a)
            _, t = self._s_t_function(in_a)
            out_b = in_b + t
            logdet = None

        if self.mask_swap:
            result = torch.cat([out_b, in_a], 1)
        else:
            result = torch.cat([in_a, out_b], 1)

        return result, logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        if self.mask_swap:
            out_a, out_b = out_b, out_a

        if self.affine:
            s, t = self._s_t_function(out_a)
            in_b = out_b / s - t  # More stable, less error   s must not equal to 0!!!
            # in_b = (out_b - t) / s
        else:
            _, t = self._s_t_function(out_a)
            in_b = out_b - t

        if self.mask_swap:
            result = torch.cat([in_b, out_a], 1)
        else:
            result = torch.cat([out_a, in_b], 1)

        return result

    def _s_t_function(self, x):
#         print('x shape is ' + str(x.shape)) 
        h = x
        for i in range(len(self.layers)-1):
            h = self.layers[i](h)
            h = self.norms[i](h)
            # h = torch.tanh(h)  # tanh may be more stable?
            h = torch.relu(h)  #
        h = self.layers[-1](h)
#         print('h shape is ' + str(h.shape))

        s = None
        if self.affine:
            # residual net for doubling the channel. Do not use residual, unstable
            log_s, t = h.chunk(2, 1)
#             print('for adj, log_s is {}, t is {}'.format(log_s.shape, t.shape))
            # s = torch.sigmoid(log_s + 2)  # (2,6,32,32) # s != 0 and t can be arbitrary : Why + 2??? more stable, keep s != 0!!! exp is not stable
            s = torch.sigmoid(log_s)  # works good when actnorm
            # s = torch.tanh(log_s) # can not use tanh
            # s = torch.sign(log_s) # lower reverse error if no actnorm, similar results when have actnorm
        else:
            t = h
        return s, t


class GraphAffineCoupling(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, masked_row, affine=True):
        super(GraphAffineCoupling, self).__init__()
        self.n_node = n_node
        self.in_dim = in_dim
        self.hidden_dim_dict = hidden_dim_dict
        self.masked_row = masked_row
        self.affine = affine

        self.hidden_dim_gnn = hidden_dim_dict['gnn']
        self.hidden_dim_linear = hidden_dim_dict['linear']

        self.net = nn.ModuleList()
        self.norm = nn.ModuleList()
        last_dim = in_dim
        for out_dim in self.hidden_dim_gnn:  # What if use only one gnn???
            self.net.append(GraphConv(last_dim, out_dim))
            self.norm.append(nn.BatchNorm1d(n_node))  # , momentum=0.9 Change norm!!!
            # self.norm.append(ActNorm2D(in_dim=n_node, logdet=False))
            last_dim = out_dim

        self.net_lin = nn.ModuleList()
        self.norm_lin = nn.ModuleList()
        for out_dim in self.hidden_dim_linear:  # What if use only one gnn???
            self.net_lin.append(GraphLinear(last_dim, out_dim))
            self.norm_lin.append(nn.BatchNorm1d(n_node))  # , momentum=0.9 Change norm!!!
            # self.norm_lin.append(ActNorm2D(in_dim=n_node, logdet=False))
            last_dim = out_dim

        if affine:
            self.net_lin.append(GraphLinear(last_dim, in_dim*2))
        else:
            self.net_lin.append(GraphLinear(last_dim, in_dim))

        self.scale = nn.Parameter(torch.zeros(1))  # nn.Parameter(torch.ones(1)) #
        mask = torch.ones(n_node, in_dim)
        mask[masked_row, :] = 0  # masked_row are kept same, and used for _s_t for updating the left rows
        self.register_buffer('mask', mask)

    def forward(self, adj, input):
#         print('input shape is '+ str(input.shape))
        masked_x = self.mask * input
#         print('masked_X shape is '+ str(masked_x.shape))
        s, t = self._s_t_function(adj, masked_x)  # s must not equal to 0!!!
        if self.affine:
            out = masked_x + (1-self.mask) * (input + t) * s
            # out = masked_x + (1-self.mask) * (input * s + t)
            logdet = torch.sum(torch.log(torch.abs(s)).view(input.shape[0], -1), 1)  # possibly wrong answer
        else:  # add coupling
            out = masked_x + t*(1-self.mask)
            logdet = None
        return out, logdet

    def reverse(self, adj, output):
        masked_y = self.mask * output
        s, t = self._s_t_function(adj, masked_y)
        if self.affine:
            input = masked_y + (1 - self.mask) * (output / s - t)
            # input = masked_x + (1 - self.mask) * ((output-t) / s)
        else:
            input = masked_y + (1 - self.mask) * (output - t)
        return input

    def _s_t_function(self, adj, x):
        # adj: (2,4,9,9)  x: # (2,9,5)
        s = None
        h = x
        for i in range(len(self.net)):
            h = self.net[i](adj, h)  # (2,1,9,hidden_dim)
            h = self.norm[i](h)
            # h = torch.tanh(h)  # tanh may be more stable
            h = torch.relu(h)  # use relu!!!

        for i in range(len(self.net_lin)-1):
            h = self.net_lin[i](h)  # (2,1,9,hidden_dim)
            h = self.norm_lin[i](h)
            # h = torch.tanh(h)
            h = torch.relu(h)

        h = self.net_lin[-1](h)
        # h =h * torch.exp(self.scale*2)
#         print('node h shape is '+ str(h.shape))
        
        if self.affine:
            log_s, t = h.chunk(2, dim=-1)
#             print('for node, log_s is {}, t is {}'.format(log_s.shape, t.shape))
            #  x = sigmoid(log_x+bias): glow code Top 1 choice, keep s away from 0, s!!!!= 0  always safe!!!
            # And get the signal from added noise in the  input
            # s = torch.sigmoid(log_s + 2)
            s = torch.sigmoid(log_s)  # better validity + actnorm

            # s = torch.tanh(log_s)  # Not stable when s =0 for synthesis data, but works well for real data in best case....
            # s = torch.sign(s)

            # s = torch.sign(log_s)

            # s = F.softplus(log_s) # negative nll
            # s = torch.sigmoid(log_s)  # little worse than +2, # *self.scale #!!! # scale leads to nan results
            # s = torch.tanh(log_s+2) # not that good
            # s = torch.relu(log_s) # nan results
            # s = log_s  # nan results
            # s = torch.exp(log_s)  # nan results
        else:
            t = h
        return s, t
class Flow(nn.Module):
    def __init__(self, in_channel, hidden_channels, affine=True, conv_lu=2, mask_swap=False):
        super(Flow, self).__init__()

        # More stable to support more flows
        self.actnorm = ActNorm(in_channel)

        if conv_lu == 0:
            self.invconv = InvConv2d(in_channel)
        elif conv_lu == 1:
            self.invconv = InvConv2dLU(in_channel)
        elif conv_lu == 2:
            self.invconv = None
        else:
            raise ValueError("conv_lu in {0,1,2}, 0:InvConv2d, 1:InvConv2dLU, 2:none-just swap to update in coupling")

        # May add more parameter to further control net in the coupling layer
        self.coupling = AffineCoupling(in_channel, hidden_channels, affine=affine, mask_swap=mask_swap)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        # out = input
        # logdet = 0
        if self.invconv:
            out, det1 = self.invconv(out)
        else:
            det1 = 0
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        if self.invconv:
            input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


# class Flow2(nn.Module): # delete
#     def __init__(self, in_channel, hidden_channels, affine=True, conv_lu=True, mask_swap=0):
#         super(Flow2, self).__init__()
#
#         # More stable to support more flows
#         self.actnorm = ActNorm(in_channel)  # Delete ActNorm first, What I need is to norm adj, rather than along batch dim
#
#         # if conv_lu:
#         #     self.invconv = InvConv2dLU(in_channel)
#         #
#         # else:
#         #     self.invconv = InvConv2d(in_channel)
#
#         # May add more parameter to further control net in the coupling layer
#         self.coupling = AffineCoupling(in_channel, hidden_channels, affine=affine, mask_swap=mask_swap)
#
#     def forward(self, input):
#         out, logdet = self.actnorm(input)
#         # out = input
#         # logdet = 0
#         # out, det1 = self.invconv(out)
#         det1 = 0
#         out, det2 = self.coupling(out)
#
#         logdet = logdet + det1
#         if det2 is not None:
#             logdet = logdet + det2
#
#         return out, logdet
#
#     def reverse(self, output):
#         input = self.coupling.reverse(output)
#         # input = self.invconv.reverse(input)
#         input = self.actnorm.reverse(input)
#
#         return input


class FlowOnGraph(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, masked_row, affine=True):
        super(FlowOnGraph, self).__init__()
        self.n_node = n_node
        self.in_dim = in_dim
        self.hidden_dim_dict = hidden_dim_dict
        self.masked_row = masked_row
        self.affine = affine
        # self.conv_lu = conv_lu
        self.actnorm = ActNorm2D(in_dim=n_node)  # May change normalization later, column norm, or row norm
        # self.invconv = InvRotationLU(n_node) # Not stable for inverse!!! delete!!!
        self.coupling = GraphAffineCoupling(n_node, in_dim, hidden_dim_dict, masked_row, affine=affine)

    def forward(self, adj, input):  # (2,4,9,9) (2,2,9,5)
        # if input are two channel identical, normalized results are 0
        # change other normalization for input
        out, logdet = self.actnorm(input)
        # out = input
        # logdet = torch.zeros(1).to(input)
        # out, det1 = self.invconv(out)
        det1 = 0
        out, det2 = self.coupling(adj, out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, adj, output):
        input = self.coupling.reverse(adj, output)
        # input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input) # change other normalization for input
        return input


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, squeeze_fold, hidden_channels, affine=True, conv_lu=2):  # in_channel: 3, n_flow: 32
        super(Block, self).__init__()
        # squeeze_fold = 3 for qm9 (bs,4,9,9), squeeze_fold = 2 for zinc (bs, 4, 38, 38)
        #                          (bs,4*3*3,3,3)                        (bs,4*2*2,19,19)
        self.squeeze_fold = squeeze_fold
        squeeze_dim = in_channel * self.squeeze_fold * self.squeeze_fold

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            if conv_lu in (0, 1):
                self.flows.append(Flow(squeeze_dim, hidden_channels,
                                       affine=affine, conv_lu=conv_lu, mask_swap=False))
            else:
                self.flows.append(Flow(squeeze_dim, hidden_channels,
                                       affine=affine, conv_lu=2, mask_swap=bool(i % 2)))

        # self.prior = ZeroConv2d(squeeze_dim, squeeze_dim*2)

    def forward(self, input):
        out = self._squeeze(input)
        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        out = self._unsqueeze(out)
        return out, logdet  # , log_p, z_new

    def reverse(self, output):  # , eps=None, reconstruct=False):
        input = self._squeeze(output)

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        unsqueezed = self._unsqueeze(input)
        return unsqueezed

    def _squeeze(self, x):
        """Trade spatial extent for channels. In forward direction, convert each
        1x4x4 volume of input into a 4x1x1 volume of output.

        Args:
            x (torch.Tensor): Input to squeeze or unsqueeze.
            reverse (bool): Reverse the operation, i.e., unsqueeze.

        Returns:
            x (torch.Tensor): Squeezed or unsqueezed tensor.
        """
        # b, c, h, w = x.size()
        assert len(x.shape) == 4
        b_size, n_channel, height, width = x.shape
        fold = self.squeeze_fold

        squeezed = x.view(b_size, n_channel, height // fold,  fold,  width // fold,  fold)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4).contiguous()
        out = squeezed.view(b_size, n_channel * fold * fold, height // fold, width // fold)
        return out

    def _unsqueeze(self, x):
        assert len(x.shape) == 4
        b_size, n_channel, height, width = x.shape
        fold = self.squeeze_fold
        unsqueezed = x.view(b_size, n_channel // (fold * fold), fold, fold, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3).contiguous()
        out = unsqueezed.view(b_size, n_channel // (fold * fold), height * fold, width * fold)
        return out


# class Block2(nn.Module): # delete
#     def __init__(self, in_channel, n_flow, squeeze_fold, hidden_channels, affine=True, conv_lu=True):  # in_channel: 3, n_flow: 32
#         super(Block2, self).__init__()
#         # squeeze_fold = 3 for qm9 (bs,4,9,9), squeeze_fold = 2 for zinc (bs, 4, 38, 38)
#         #                          (bs,4*3*3,3,3)                        (bs,4*2*2,19,19)
#         self.squeeze_fold = squeeze_fold
#         squeeze_dim = in_channel * self.squeeze_fold * self.squeeze_fold
#
#         self.flows = nn.ModuleList()
#         for i in range(n_flow):
#             self.flows.append(Flow2(squeeze_dim, hidden_channels, affine=affine, conv_lu=conv_lu, mask_type=i % 2))
#
#         self.prior = ZeroConv2d(squeeze_dim, squeeze_dim*2)
#
#     def forward(self, input):
#         out = self._squeeze(input)
#         logdet = 0
#
#         for flow in self.flows:
#             out, det = flow(out)
#             logdet = logdet + det
#
#         out = self._unsqueeze(out)
#         return out, logdet  # , log_p, z_new
#
#     def reverse(self, output):  # , eps=None, reconstruct=False):
#         input = self._squeeze(output)
#
#         for flow in self.flows[::-1]:
#             input = flow.reverse(input)
#
#         unsqueezed = self._unsqueeze(input)
#         return unsqueezed
#
#     def _squeeze(self, x):
#         """Trade spatial extent for channels. In forward direction, convert each
#         1x4x4 volume of input into a 4x1x1 volume of output.
#
#         Args:
#             x (torch.Tensor): Input to squeeze or unsqueeze.
#             reverse (bool): Reverse the operation, i.e., unsqueeze.
#
#         Returns:
#             x (torch.Tensor): Squeezed or unsqueezed tensor.
#         """
#         # b, c, h, w = x.size()
#         assert len(x.shape) == 4
#         b_size, n_channel, height, width = x.shape
#         fold = self.squeeze_fold
#
#         squeezed = x.view(b_size, n_channel, height // fold,  fold,  width // fold,  fold)
#         squeezed = squeezed.permute(0, 1, 3, 5, 2, 4).contiguous()
#         out = squeezed.view(b_size, n_channel * fold * fold, height // fold, width // fold)
#         return out
#
#     def _unsqueeze(self, x):
#         assert len(x.shape) == 4
#         b_size, n_channel, height, width = x.shape
#         fold = self.squeeze_fold
#         unsqueezed = x.view(b_size, n_channel // (fold * fold), fold, fold, height, width)
#         unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3).contiguous()
#         out = unsqueezed.view(b_size, n_channel // (fold * fold), height * fold, width * fold)
#         return out


class BlockOnGraph(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, n_flow, mask_row_size=1, mask_row_stride=1, affine=True):  #, conv_lu=True):
        """

        :param n_node:
        :param in_dim:
        :param hidden_dim:
        :param n_flow:
        :param mask_row_size: number of rows to be masked for update
        :param mask_row_stride: number of steps between two masks' firs row
        :param affine:
        """
        # in_channel=2 deleted. in_channel: 3, n_flow: 32
        super(BlockOnGraph, self).__init__()
        assert 0 < mask_row_size < n_node
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            start = i * mask_row_stride
            masked_row =[r % n_node for r in range(start, start+mask_row_size)]
            self.flows.append(FlowOnGraph(n_node, in_dim, hidden_dim_dict, masked_row=masked_row, affine=affine))
        # self.prior = ZeroConv2d(2, 4)

    def forward(self, adj, input):
        out = input
        logdet = 0
        for flow in self.flows:
            out, det = flow(adj, out)
            logdet = logdet + det
            # it seems logdet is not influenced
        return out, logdet

    def reverse(self, adj, output):
        input = output
        for flow in self.flows[::-1]:
            input = flow.reverse(adj, input)
        return input


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, squeeze_fold, hidden_channel, affine=True, conv_lu=2): # in_channel: 3, n_flow:32, n_block:4
        super(Glow, self).__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel  # 3
        for i in range(n_block):
            self.blocks.append(Block(n_channel, n_flow, squeeze_fold, hidden_channel, affine=affine, conv_lu=conv_lu)) # 3,6,12
            # self.blocks.append(
            #     Block2(n_channel, n_flow, squeeze_fold, hidden_channel, affine=affine, conv_lu=conv_lu))  # delete

    def forward(self, input):
        logdet = 0
        out = input

        for block in self.blocks:
            out, det = block(out)
            logdet = logdet + det

        return out, logdet

    def reverse(self, z):  # _list, reconstruct=False):
        h = z
        for i, block in enumerate(self.blocks[::-1]):
            h = block.reverse(h)

        return h


class GlowOnGraph(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, n_flow, n_block,
                 mask_row_size_list=[2], mask_row_stride_list=[1], affine=True):  # , conv_lu=True): # in_channel: 2 default
        super(GlowOnGraph, self).__init__()

        assert len(mask_row_size_list) == n_block or len(mask_row_size_list) == 1
        assert len(mask_row_stride_list) == n_block or len(mask_row_stride_list) == 1
        if len(mask_row_size_list) == 1:
            mask_row_size_list = mask_row_size_list * n_block
        if len(mask_row_stride_list) == 1:
            mask_row_stride_list = mask_row_stride_list * n_block
        self.blocks = nn.ModuleList()
        for i in range(n_block):
            mask_row_size = mask_row_size_list[i]
            mask_row_stride = mask_row_stride_list[i]
            self.blocks.append(BlockOnGraph(n_node, in_dim, hidden_dim_dict, n_flow, mask_row_size, mask_row_stride, affine=affine))

    def forward(self, adj, x):
        # adj (bs, 4,9,9), xx:(bs, 9,5)
        logdet = 0
        out = x
        for block in self.blocks:
            out, det = block(adj, out)
            logdet = logdet + det

        return out, logdet

    def reverse(self, adj, z):
        # (bs, 4,9,9), zz: (bs, 9, 5)
        input = z
        for i, block in enumerate(self.blocks[::-1]):
            input = block.reverse(adj, input)

        return input
