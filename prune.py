from parameter import get_parameter
import torch
from models.HED3 import HED

import yaml
import argparse 
import random
import numpy

from attrdict import AttrDict


def prune_network(args, network):
    device = torch.device("cuda" if args.gpu_no >= 0 else "cpu")
    device = torch.device("cpu")

    # prune network
    network = prune_step(network, args.prune_layers,
                         args.prune_channels, args.independent_prune_flag)
    network = network.to(device)
    if args.verbose:
        print("-*-"*10 + "\n\tPrune network\n" + "-*-"*10)
    # print(network)

    return network


def prune_step(net, prune_layers, prune_channels, independent_prune_flag):
    net = net.cpu()

    count = 0  # count for indexing 'prune_channels'
    conv_count = 1  # conv count for 'indexing_prune_layers'
    # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    dim = 0
    residue = None  # residue is need to prune by 'independent strategy'

    channel_index_dict = dict()
    for name, layer in net.named_modules():
        #print("Out", name, layer)
        """
        for name, layer in seq.named_modules():
            if name == '':
                continue
            print("In", name, layer)
            name = int(name)
        """
        if isinstance(layer, torch.nn.Conv2d):
            lst = name.split(".")
            if len(lst) < 2:
                name = lst[0]
                idx = None
                #print("conv:", net._modules[name])

                def prune_score_layer(score_layer_name, conv_layer_name):
                    if name == score_layer_name and conv_layer_name in channel_index_dict:
                        # print(net._modules[name].weight.data.shape)
                        channel_index_dict[conv_layer_name].sort()
                        lst = channel_index_dict[conv_layer_name]

                        conv = net._modules[name]

                        new_conv = torch.nn.Conv2d(in_channels=conv.in_channels-len(lst),
                                                   out_channels=1,
                                                   kernel_size=conv.kernel_size,
                                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
                        #print("new conv", new_conv.weight.data.shape)
                        channel_index_lst_need = set(
                            list(range(conv.in_channels))).difference(lst)

                        count = 0
                        for i in channel_index_lst_need:
                            new_conv.weight.data[:,
                                                 count] = conv.weight.data[:, i]
                            count += 1
                        net._modules[name] = new_conv
                prune_score_layer('dsn1', 'conv2')
                prune_score_layer('dsn2', 'conv4')
                prune_score_layer('dsn3', 'conv7')

            else:
                name = lst[0]
                idx = int(lst[1])
                #print("conv:", net._modules[name][idx])

                if dim == 1:
                    new_, residue = get_new_conv(
                        net._modules[name][idx], dim, channel_index, independent_prune_flag)
                    net._modules[name][idx] = new_
                    dim ^= 1

                if 'conv%d' % conv_count in prune_layers:
                    channel_index = get_channel_index(
                        net._modules[name][idx].weight.data, prune_channels[count], residue)

                    channel_index_dict['conv%d' % conv_count] = channel_index
                    new_ = get_new_conv(
                        net._modules[name][idx], dim, channel_index, independent_prune_flag)
                    net._modules[name][idx] = new_
                    dim ^= 1
                    count += 1
                else:
                    residue = None

                conv_count += 1

    return net



def get_channel_index(kernel, num_elimination, residue=None):
    # get cadidate channel index for pruning
    # 'residue' is needed for pruning by 'independent strategy'

    sum_of_kernel = torch.sum(
        torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
    if residue is not None:
        sum_of_kernel += torch.sum(
            torch.abs(residue.view(residue.size(0), -1)), dim=1)

    vals, args = torch.sort(sum_of_kernel)

    return args[:num_elimination].tolist()


def index_remove(tensor, dim, index, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())
    new_size = tensor.size(dim) - len(index)
    size_[dim] = new_size
    new_size = size_

    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))

    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor


def get_new_conv(conv, dim, channel_index, independent_prune_flag=False):
    if dim == 0:
        new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                                   out_channels=int(
                                       conv.out_channels - len(channel_index)),
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

        new_conv.weight.data = index_remove(
            conv.weight.data, dim, channel_index)
        new_conv.bias.data = index_remove(conv.bias.data, dim, channel_index)

        return new_conv

    elif dim == 1:
        new_conv = torch.nn.Conv2d(in_channels=int(conv.in_channels - len(channel_index)),
                                   out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

        new_weight = index_remove(
            conv.weight.data, dim, channel_index, independent_prune_flag)
        residue = None
        if independent_prune_flag:
            new_weight, residue = new_weight
        new_conv.weight.data = new_weight
        new_conv.bias.data = conv.bias.data

        return new_conv, residue


def get_new_norm(norm, channel_index):
    new_norm = torch.nn.BatchNorm2d(num_features=int(norm.num_features - len(channel_index)),
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)

    new_norm.weight.data = index_remove(norm.weight.data, 0, channel_index)
    new_norm.bias.data = index_remove(norm.bias.data, 0, channel_index)

    if norm.track_running_stats:
        new_norm.running_mean.data = index_remove(
            norm.running_mean.data, 0, channel_index)
        new_norm.running_var.data = index_remove(
            norm.running_var.data, 0, channel_index)

    return new_norm


def get_new_linear(linear, channel_index):
    new_linear = torch.nn.Linear(in_features=int(linear.in_features - len(channel_index)),
                                 out_features=linear.out_features,
                                 bias=linear.bias is not None)
    new_linear.weight.data = index_remove(linear.weight.data, 1, channel_index)
    new_linear.bias.data = linear.bias.data

    return new_linear


if __name__ == '__main__':
    args = get_parameter()

    #path = "/home/hpc/nctu/hed/ckpt/standard/log/new_github_adam_lr1e-4_Aug_0.3_savemodel_Feb25_12-13-51/epoch_0.pth"
    #path = "/home/hpc/nctu/hed/ckpt/no_norm/log/adam_lr1e-4_Aug_0.3_savemodel_Mar02_14-00-23/epoch_0.pth"
    path = "/home/hpc/nctu/hed/ckpt/no_norm/log/adam_lr1e-4_Aug_0.3_savemodel_Mar02_16-17-56/epoch_1.pth"
    cfg = None
    #cfg_name = 'standard.yaml'
    cfg_name = 'no_norm.yaml'
    with open('config/' + cfg_name, 'r') as f:
        cfg = AttrDict( yaml.load(f) )

    net = HED(cfg)
    net.load_state_dict(torch.load(path))
    net.eval()
    print(net)

    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    net = prune_network(args, net)
    print(net)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    #torch.save(net, "out.pth")

    with torch.no_grad():
        traced_cell = torch.jit.trace(net, torch.FloatTensor(
            torch.rand([1, 3, 120, 160])))
    torch.jit.save(traced_cell, "pruned.pt")
