import json
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from utils.Setup import trans_to_cuda
import os


def create_dataloaders(args):
    train_dataset = CascadeData(args, args.cascade_train_path)
    val_dataset = CascadeData(args, args.cascade_valid_path)
    test_dataset = CascadeData(args, args.cascade_test_path)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    test_sampler = SequentialSampler(test_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                sampler=val_sampler,
                                pin_memory=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.test_batch_size,
                                 sampler=test_sampler,
                                 pin_memory=True)
    return train_dataloader, val_dataloader, test_dataloader


class CascadeData(Dataset):
    def __init__(self, args, dataPath):
        self.max_len = args.max_len
        self.EOS = args.user_num - 1
        self.seed = args.seed

        with open(dataPath, 'r') as cas_file:
            self.cascade_data = json.load(cas_file)

    def __len__(self) -> int:
        return len(self.cascade_data)

    def __getitem__(self, idx: int) -> dict:
        # Following the previous works, we also predict the end of cascade.
        cascade = self.cascade_data[idx]['cascade'] + [self.EOS]
        cas_raw = cascade[:self.max_len + 1] if len(cascade) > self.max_len + 1 else cascade

        cas_raw = torch.Tensor(cas_raw).long().squeeze()
        cascade = pad_1d_tensor(cas_raw, self.max_len + 1)

        data = dict(
            cascade=cascade
        )
        return data


def dataProcess(args, data):
    cas_pad = data['cascade']
    cascade = trans_to_cuda(cas_pad[:, :-1])
    cas_mask = trans_to_cuda((cascade == 0))
    label = trans_to_cuda(cas_pad[:, 1:]).contiguous().view(-1)
    label_mask = trans_to_cuda(get_previous_user_mask(cascade, args.user_num))

    return cascade, cas_mask, label, label_mask


def RawGraph(args):
    """
    导入社交网络图
    """
    friend_net = args.graphPath
    edges = []
    if os.path.exists(friend_net):
        with open(friend_net, 'r') as handle:
            friend_in = handle.read().strip().split("\n")
            friend_in = [edge.split(',') for edge in friend_in]

            friend_in = [(int(edge[0]), int(edge[1])) for edge in friend_in]
        edges += friend_in
    else:
        return edges

    edges_list_tensor = torch.LongTensor(edges).t()
    edges_weight = torch.ones(edges_list_tensor.size(1)).float()
    data = Data(edge_index=edges_list_tensor, edge_attr=edges_weight)

    return data


#################################################
# 后面的代码应该是不需要管的
#################################################
def get_previous_user_mask(seq, user_size):
    """ Mask previous activated users."""
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    return masked_seq


def pad_1d_tensor(tensor, max_len, pad_value=0):
    len_seq = len(tensor)

    if len_seq < max_len:
        pad_len = max_len - len_seq
        padded_tensor = F.pad(tensor, (0, pad_len), value=pad_value)
    else:
        padded_tensor = tensor[:max_len]
    return padded_tensor


def pad_2d_tensor(tensor, max_len, pad_value=0):
    tensor = tensor.long().squeeze()
    len_seq, dim = tensor.size()

    if len_seq < max_len:
        pad_len = max_len - len_seq
        padded_tensor = F.pad(tensor, (0, 0, 0, pad_len), value=pad_value)
    else:
        padded_tensor = tensor[:max_len, :]
    return padded_tensor
