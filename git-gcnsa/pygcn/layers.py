import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from distcalculate import consinesimilar, rbfkernel
import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
GPU_VISIBLE_DEVICES = 0,1,2,3

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Fulconnect(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Fulconnect, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class dotproduct(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features):
        super(dotproduct, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.FloatTensor(1, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mul(input, self.weight)
        return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' ')'

class adjcentmatrix(Module):
    def __init__(self, in_features, epsilon, n_samples, nhid, dropout1,r):
        super(adjcentmatrix, self).__init__()
        self.n_head = 4
        self.in_features = in_features
        self.dropout1 = dropout1
        self.nhid = nhid
        self.r = r
        self.out_features = 8
        self.out_features = 8
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features*self.n_head))
        self.weight2 = Parameter(torch.FloatTensor(self.in_features, self.out_features*self.n_head))
        self.weight3 = Parameter(torch.FloatTensor(self.in_features, self.nhid))
        self.reset_parameters()
        self.eye_matrix = torch.eye(self.n_samples).cuda()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight3.size(1))
        self.weight3.data.uniform_(-stdv, stdv)

    def forward(self, input):
        attention2 = torch.zeros(self.n_samples, self.n_samples)
        attention2 = attention2.cuda()
        zero_matrix = attention2
        inter_put1 = torch.mm(input, self.weight1)
        inter_put2 = torch.mm(input, self.weight2)
        inter_put3 = torch.mm(input, self.weight3)
        j=0
        adjacent = zero_matrix.cuda
        for i in range(self.n_head):
            input1 = inter_put1[:,j:j+self.out_features]
            input2 = inter_put2[:,j:j+self.out_features]
            adjacent = + consinesimilar(input1, input2)
            j = j + self.out_features
        #adjacent = torch.clamp(adjacent, min=0)

        attention1 = torch.where(adjacent > self.epsilon, adjacent, zero_matrix)

        attention2 = zero_matrix
        b = torch.topk(adjacent, k=self.r, dim=1, largest=True)
        attention2 = attention2.scatter_(1, b.indices, b.values)

        attention = attention1+attention2
        attention = torch.where(attention> 0, adjacent, zero_matrix)
        f = torch.isnan(attention)
        attention[f] = 0

        attention = torch.where(attention > attention.t(), attention,attention.t())

        attention = attention + torch.eye(self.n_samples).cuda()
        #DEBUG
        #print("⚠️ Before normalize_adj:")
        #print("  sum per row (min/max):", attention.sum(1).min().item(), attention.sum(1).max().item())

        attention = normalize_adj(attention)
        
        #print("✅ After normalize_adj:")
        #print("  has NaNs:", torch.isnan(attention).any().item())
        #print("  attention row sums (min/max):", attention.sum(1).min().item(), attention.sum(1).max().item())
        f = torch.isnan(attention)
        attention[f] = 0
        output = torch.mm(attention, inter_put3)
        output = F.dropout(output, self.dropout1, training=self.training)
        return output,attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.n_samples) + ')'

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):   # [1966,4]

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads  # 491.5
        self.dropout = 0.5
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim  #1966
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)  # [1966,491]*4
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)  # [1966,491]*4
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)  # [1966,491]*4
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)  # [1966,491]*4

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)  # [256,1966]->[256,1964]->[256,4,491]->[4,256,625]
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)  # [256,1966]->[256,1964]->[256,4,491]->[4,256,625]
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)  # [256,1966]->[256,1964]->[256,4,491]->[4,256,625]

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # [4,256,491]*[4,491,256]->[4,256,256]
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [4,256,256]*[4,256,625]->[4,256,491]
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        # context = self.fc(context)
        # context = F.dropout(context, self.dropout, training=self.training)
        return context

class EncoderLayer1(torch.nn.Module):
    def __init__(self, input_dim, n_heads, dropout):  # [1966,4]
        super(EncoderLayer1, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)   # [1966,4]
        self.AN1 = torch.nn.LayerNorm(input_dim)
        self.dropout = dropout
        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = F.relu(self.attn(X)) # 多头注意力去聚合其他DDI的特
        output = F.dropout(output, self.dropout, training=self.training)
        output = output + X # 残差连接+LayerNorm
        # #
        output = F.relu(output) # FC
        output = F.dropout(output, self.dropout, training=self.training)
        output = output + X # 残差连接+LayerNorm
        # output = F.dropout(output, self.dropout, training=self.training)

        return output


# len_after_AE = 500
bert_n_heads =4
drop_out_rating = 0.5


def normalize_adj(input):
    """Row-normalize sparse matrix"""
    sum_input = torch.sum(input, dim=1)
    degree_input = torch.diag(sum_input.pow(-0.5))
    input_norm = torch.mm(degree_input, torch.mm(input, degree_input))
    return input_norm

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))