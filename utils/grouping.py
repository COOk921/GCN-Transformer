import torch
import pdb

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