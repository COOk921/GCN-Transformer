import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
from dataLoader import dataProcess
from module import GraphEncoder, CascadeEncoder
from utils.grouping import div_user_group,grouping
import pdb

class SILN(nn.Module):
    def __init__(self, args):
        super(SILN, self).__init__()
        self.dim = args.dim
        self.user_num = args.user_num

        self.Embed = nn.Embedding(self.user_num, self.dim, padding_idx=0)  # 初始化用户嵌入
        self.GNN = GraphEncoder(args)
        self.SAN = CascadeEncoder(args)

        #self.reduce_channels = nn.MaxPool1d(kernel_size=args.group_num)  #Maxpooling
        self.reduce_channels = nn.AvgPool1d(kernel_size=args.group_num)  

        self.Predict = nn.Linear(self.dim, self.user_num)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, args, data, graph):
        cascade, cas_mask, label, label_mask = dataProcess(args, data)

        # # 社交网络的图卷积
        initial_user = self.Embed.weight  # 可更新的初始嵌入（因为数据集里没有node feature）
        user_Embeddings = self.GNN(initial_user, graph)

        casEmbed = F.embedding(cascade, user_Embeddings)
        # divide to groups
        groupEmbed, group_mask, count = grouping(casEmbed, cas_mask, args)        
        h = self.SAN(groupEmbed ,groupEmbed, groupEmbed, group_mask, count) #Q:groupEmbed, K:casEmbed, V:casEmbed

        # # channel reduction
        #h = self.reduce_channels(h.transpose(1, 2)).squeeze(-1)   
        h = self.reduce_channels(h.transpose(1, 2)).squeeze(-1)   
       
        # # 预测
        pred_user = h@user_Embeddings.t() + label_mask  
        pred_user = self.sigmoid(pred_user)

        return pred_user, label
