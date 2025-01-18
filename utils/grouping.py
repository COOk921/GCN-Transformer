import torch
import pdb
import numpy as np
import time
from sklearn.cluster import KMeans,Birch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from utils.Setup import trans_to_cuda


#平均分组
def div_user_group(casEmbed, cas_mask, args):

    usertoGroup = [torch.arange(batch.size(0)).unsqueeze(0) // args.group_size for batch in casEmbed] 
    casEmbed = casEmbed * ( ~cas_mask.unsqueeze(-1) )

    groupEmbed = casEmbed.view(
        casEmbed.size(0),
        casEmbed.size(1) // args.group_size, 
        args.group_size,
        casEmbed.size(2)
    ).sum(dim=2) 

    group_mask = cas_mask.view(
        cas_mask.size(0),
        cas_mask.size(1) // args.group_size,
        args.group_size
    ).all(dim=2)

    return groupEmbed, group_mask

# 使用密度聚类分组

def grouping(casEmbed, cas_mask ,args):

    batch_size, seq_len, embed_size = casEmbed.size()
    casEmbed = casEmbed * ( ~cas_mask.unsqueeze(-1) ) 

    all_labels = []
    for b in range(batch_size):
        batch_data = casEmbed[b]
        batch_mask = cas_mask[b] 

        max_len = torch.sum(~batch_mask).item() # 有效长度
        k = max_len // args.group_size  
        k = 2 if k < 2 else k          # 至少要分两组
    
        labels_b = cluster_with_Birch(batch_data, k)
        
        all_labels.append(labels_b)

    grouped_labels = torch.stack(all_labels, dim=0)
    grouped_labels = trans_to_cuda(grouped_labels)

    num_groups = torch.max(grouped_labels).item() + 1
    
    groupEmbed = torch.zeros(
        batch_size, 
        num_groups, 
        embed_size, 
        device=casEmbed.device
    )
   
    index = grouped_labels.unsqueeze(-1).expand(-1, -1,embed_size) 
    groupEmbed.scatter_add_(1, index, casEmbed)

    groupEmbed = torch.nn.functional.pad(groupEmbed, (0, 0, 0, args.group_num - num_groups))
    group_mask = torch.all(groupEmbed == 0, dim=2)

    return groupEmbed, group_mask

def cluster_with_Birch(X, k ):

    X_cpu = X.cpu().detach().numpy()
    # MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_norm = scaler.fit_transform(X_cpu)

    # PCA
    # pca = PCA(n_components = X.size(-1)//4  ) # 64 -> 8
    # batch_pca = pca.fit_transform(X_cpu)

    birch = Birch(n_clusters=k,threshold = 0.3)
    birch.fit(X_norm)

    labels = torch.from_numpy(birch.labels_)

    return labels



