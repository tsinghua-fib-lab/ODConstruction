import sys
import copy
import time

import numpy as np
import torch.multiprocessing as mp
# import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as fct
from torch.autograd import Variable, Function, gradcheck
from torch.nn.utils import weight_norm

from torchviz import make_dot

import dgl.function as fn
from dgl.nn.pytorch import GATConv

from utils import *



class GAT(nn.Module):
    def __init__(self, config):
        super(GAT, self).__init__()
        self.config = config

        self.activation = getattr(fct, config["activation"])

        self.gat_layers = nn.ModuleList()

        # input projection (no residual)
        self.gat_layers.append(
            GATConv(config["num_in"], config["num_hidden"], config["GATheads"][0],
            config["feat_drop"], config["attn_drop"], config["negative_slope"], False, self.activation))

        # hidden layers
        for l in range(1, config["num_layers"]):
            # due to multi-head, the num_in = num_hidden * num_heads
            self.gat_layers.append(
                GATConv(config["num_hidden"] * config["GATheads"][l-1], config["num_hidden"], config["GATheads"][l],
                config["feat_drop"], config["attn_drop"], config["negative_slope"], config["residual"], self.activation))

        # output projection
        self.gat_layers.append(
            GATConv(config["num_hidden"] * config["GATheads"][-2], config["num_out"], config["GATheads"][-1],
            config["feat_drop"], config["attn_drop"], config["negative_slope"], config["residual"], None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.config["num_layers"]):
            h = self.gat_layers[l](self.config["g"], h).flatten(1)
        # output projection
        embeddings = self.gat_layers[-1](self.config["g"], h).mean(1)

        return embeddings

class Gravity_inspired_link_predictor(nn.Module):
    '''
    Reference to paper, Gravity-Inspired Graph Autoencoders for Directed Link Prediction
    '''
    def __init__(self, config):
        super(Gravity_inspired_link_predictor, self).__init__()

        self.G = nn.Parameter(torch.FloatTensor(1))
        self.beta1 = nn.Parameter(torch.FloatTensor(1))
        self.beta2 = nn.Parameter(torch.FloatTensor(1))
        self.Lambda = nn.Parameter(torch.FloatTensor(1))

    def forward(self, origin_embedding, destination_embedding):

        M_origin = origin_embedding[0]
        M_destination = destination_embedding[0]
        L_origin = origin_embedding[1:]
        L_destination = destination_embedding[1:]

        r = torch.sqrt(torch.sum((L_origin - L_destination) **2))

        # F = self.G * () / ()

        A = torch.relu(self.beta1 * M_origin + self.beta2 * M_destination - self.Lambda * torch.log(r))

        return A

class GraphConstructor(nn.Module):

    def __init__(self, config):
        super(GraphConstructor, self).__init__()
        self.config = config

        self.gnn = GAT(config)

        # self.bilinear_predictor = nn.Linear(config["num_out"], config["num_out"])

        self.linear_predictor = nn.Linear(config["num_out"] * 2 + 1, 1)

        # gravity Inspired
        self.lambda1 = nn.Parameter(torch.FloatTensor(1))
        self.lambda2 = nn.Parameter(torch.FloatTensor(1))
        self.lambda3 = nn.Parameter(torch.FloatTensor(1))
        self.beta = nn.Parameter(torch.FloatTensor(1))
        self.G = nn.Parameter(torch.FloatTensor(1))

        nn.init.uniform_(self.lambda1, a=0.0, b=1.0)
        nn.init.uniform_(self.lambda2, a=0.0, b=1.0)
        nn.init.uniform_(self.lambda3, a=0.0, b=1.0)
        nn.init.uniform_(self.G, a=0.0, b=1.0)

        # self.gravty_predictor = Gravity_inspired_link_predictor(config)

    def Graph_embedding(self, node_feats):
        node_embedding = self.gnn(node_feats)
        return node_embedding

    def gravity_prediction(self, embeddings, distance):
        # print("GNN", embeddings.mean())
        embeddings1 = embeddings
        embeddings2 = embeddings

        M1 = embeddings1[:, 0].unsqueeze(dim=0).repeat(embeddings1.size(0), 1)
        M2 = embeddings2[:, 0].unsqueeze(dim=1).repeat(1, embeddings2.size(0))

        L1 = embeddings1[:, 1:].unsqueeze(dim=0).repeat(embeddings1.size(0), 1, 1)
        L2 = embeddings2[:, 1:].unsqueeze(dim=1).repeat(1, embeddings2.size(0), 1)
        r = torch.mean((L1 - L2) **2, dim=2)
        r = r + self.beta * distance
        r = torch.clamp(r, min = 1e-7)
        r = torch.sqrt(r)

        M1, M2 = torch.abs(M1), torch.abs(M2)

        # OD = self.G * ( (M1 * M2) / r )

        OD = self.G * ( (M1 **self.lambda1) * (M2 **self.lambda2) / (r **self.lambda3) )
        for i in range(OD.size(0)):
            OD[i, i] = 0

        return OD

    def dot_prediction(self, embeddings):
        embeddings1 = embeddings
        embeddings2 = embeddings.transpose(0, 1)
        OD = embeddings1.matmul(embeddings2)
        return OD

    def bilinear_prediction(self, embeddings):
        embeddings1 = embeddings
        embeddings2 = embeddings.transpose(0, 1)
        OD = self.bilinear_predictor(embeddings1).matmul(embeddings2)
        return OD

    def linear_prediction(self, embeddings, distance):
        embeddings1 = embeddings.unsqueeze(dim=0).repeat(embeddings.size(0), 1, 1)
        embeddings2 = embeddings.unsqueeze(dim=1).repeat(1, embeddings.size(0), 1)

        pair_emb = torch.cat((embeddings1, embeddings2), dim=2)
        OD = torch.relu(self.linear_predictor(pair_emb).squeeze())
        for i in range(OD.size(0)):
            OD[i, i] = 0
        return OD

    def forward(self, region_attributes, distance):
        embeddings = self.Graph_embedding(region_attributes)
        adjacency = self.gravity_prediction(embeddings, distance)
        # adjacency = self.linear_prediction(embeddings)
        return adjacency



'''
Conditional Wasserstain GAN
'''
class Generator(nn.Module):


    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config

        self.g_constructor = GraphConstructor(config)
        self.threshold = nn.Threshold(threshold = 1, value = 0)

    def generate_OD_net(self, region_attributes, distance):
        self.OD_net = self.g_constructor(region_attributes, distance)
        self.adjacency = self.threshold(self.OD_net)
        self.logp = self.OD_net / (self.OD_net.sum(1).unsqueeze(dim=1) + 1e-10)
        self.logp = torch.log(self.logp + 1e-10)

    def sample_a_neighbor(self, node):
        o_idx = torch.argmax(node)

        node = node.unsqueeze(dim=1)[o_idx].repeat(node.size(0))
        neighbors = node * self.logp[o_idx]
        next_node = fct.gumbel_softmax(neighbors, tau = self.config["temperature"], hard=True)
        return next_node

    def one_hot_to_nodefeat(self, node, all_feats):
        node_idx = torch.argmax(node)

        feats = all_feats[:,:-self.config["noise_dim"]]
        node = node.unsqueeze(dim=1)[node_idx].repeat(feats.size(1))
        feat = feats[node_idx] * node
        return feat

    def sample_one_random_walk(self, region_attributes, distance):
        node_seq = []
        feat_seq = []
        edge_seq = []
        dis_seq = []

        init_node = torch.randint(low = 0, high = self.adjacency.size(0), size = (1,))[0].to(self.config["device"])
        init_node = fct.one_hot(init_node, num_classes = self.adjacency.size(0))

        node_seq.append(init_node)
        feat_seq.append(self.one_hot_to_nodefeat(init_node, region_attributes))

        for i in range(self.config["len_random_walk"] -1):

            next_node = self.sample_a_neighbor(node_seq[-1])
            feat = self.one_hot_to_nodefeat(next_node, region_attributes)
            flow_between = self.adjacency[torch.argmax(node_seq[-1]), torch.argmax(next_node)]
            dis_between = distance[torch.argmax(node_seq[-1]), torch.argmax(next_node)]

            node_seq.append(next_node)
            feat_seq.append(feat)
            edge_seq.append(flow_between)
            dis_seq.append(dis_between)

        # node_seq = torch.stack(node_seq[1:])
        feat_seq = torch.stack(feat_seq[1:])
        edge_seq = torch.stack(edge_seq).view([-1, 1])
        dis_seq = torch.stack(dis_seq).view([-1, 1])
        # seq = torch.cat((node_seq, edge_seq, dis_seq), dim=1) # node_seq, feat_seq, edge_seq, dis_seq
        seq = edge_seq
        return seq


    def sample_generated_batch(self, region_attributes, distance, batch_size):
        self.generate_OD_net(region_attributes, distance)

        # # start_time = time.time()
        # batch = []
        # for i in range(batch_size):
        #     batch.append(self.sample_one_random_walk(region_attributes, distance))
        # batch = torch.stack(batch)
        # # print("walk sample time cost:", time.time() - start_time)

        start_time = time.time()
        batch = []
        for i in range(batch_size):
            batch.append(sample_one_random_walk(self.adjacency, self.logp, region_attributes, distance, self.config))
        batch = torch.stack(batch)
        print("walk sample time cost:", time.time() - start_time)



        start_time = time.time()
        batch = []
        mp.set_start_method('forkserver')
        processes = []
        for i in range(batch_size):
            p = mp.Process(target = sample_one_random_walk, args=(self.adjacency, self.logp, region_attributes, distance, self.config,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        # batch = torch.stack(batch)
        print("walk sample time cost:", time.time() - start_time)
        exit(0)

        return batch


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=5, dropout=0.05):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class Discriminator(nn.Module):
    '''
    Self-attention based binary classifier
    '''

    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config

        # ## self-attention
        # self.fc_in = nn.Linear(config["random_walk_dim"] - 1, config["seq_indim"] - 1)
        #
        # self.seq_process = nn.MultiheadAttention(embed_dim = config["seq_indim"], \
        #                                          num_heads = config["seq_heads"], \
        #                                          add_bias_kv=True, \
        #                                          dropout = config["seq_dropout"], \
        #                                          batch_first=True)
        #
        # self.fc_out = nn.Linear(64, 1)
        #
        # self.fc_pre = nn.Linear((config["len_random_walk"] -1) * config["seq_indim"], 1)

        # TCN


        # ## GRU
        # self.fc_in = nn.Linear(config["random_walk_dim"], config["seq_indim"])
        #
        # self.rnn = nn.GRU(input_size = config["seq_indim"], \
        #                   hidden_size = config["seq_hiddim"], \
        #                   num_layers = config["seq_layers"], \
        #                   batch_first=True)
        #
        # self.fc_pre = nn.Linear(config["seq_layers"] * config["seq_hiddim"], 1)


        # TemporalConvNet
        self.fc_in = nn.Linear(config["random_walk_dim"], config["seq_indim"])
        self.tcn = TemporalConvNet(num_inputs = config["seq_indim"], num_channels = [config["seq_hiddim"]]*7)
        self.fc_pre = nn.Linear(config["seq_hiddim"], 1)

    def forward(self, x):

        # self-attention
        # x_node = x[:, :, :self.config["random_walk_dim"]-1]
        # x_edge = x[:, :, self.config["random_walk_dim"]-1:]
        #
        # x_node = self.fc_in(x_node)
        # x = torch.cat((x_node, x_edge), dim=2)
        # x, _ = self.seq_process(x, x, x)
        # x = x.reshape([x.size(0), -1])
        # # x = self.fc_out(x).squeeze()
        # pre = self.fc_pre(x).squeeze()


        # # RNN
        # x = self.fc_in(x)
        # with torch.backends.cudnn.flags(enabled=False):
        #     out, hn = self.rnn(x)
        #
        # x = hn.transpose(0, 1).reshape([x.size(0), -1])
        #
        # pre = self.fc_pre(x).squeeze()


        # TCN
        x = self.fc_in(x).transpose(1, 2)
        x = self.tcn(x)
        pre = self.fc_pre(x[:, :, -1]).squeeze()

        return pre
