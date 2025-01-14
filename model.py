import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
from dataLoader import dataProcess
from module import GraphEncoder, CascadeEncoder


class SILN(nn.Module):
    def __init__(self, args):
        super(SILN, self).__init__()
        self.dim = args.dim
        self.user_num = args.user_num

        self.Embed = nn.Embedding(self.user_num, self.dim, padding_idx=0)  # 初始化用户嵌入
        self.GNN = GraphEncoder(args)
        self.SAN = CascadeEncoder(args)

        self.Predict = nn.Linear(self.dim, self.user_num)
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

        # # 抽取参与者的embedding并使用Trm建模cascade
        casEmbed = F.embedding(cascade, user_Embeddings)
        h = self.SAN(casEmbed, cas_mask)

        # # 预测
        pred_user = self.Predict(h) + label_mask
        pred_user = pred_user.view(-1, pred_user.size(-1))

        return pred_user, label
