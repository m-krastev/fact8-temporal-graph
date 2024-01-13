# """
# Extended GAT for handling timestamps on edges.

# """
# import torch
# import torch.nn as nn
# from torch_geometric.nn import GATConv, GATv2Conv
# import numpy as np

# from tgnnexplainer.xgraph.models.ext.tgat.graph import NeighborFinder


# class GATStatic(nn.Module):
#     def __init__(self, ngh_finder: NeighborFinder, n_feat, e_feat,
#                  attn_mode='prod', use_time='time', agg_method='attn',
#                  num_layers=2, n_head=2, null_idx=0, drop_out=0.1, seq_len=None):
#         super(GATStatic, self).__init__()

#         self.ngh_finder = ngh_finder
#         self.n_feat = n_feat
#         self.e_feat = e_feat
#         self.layers = [None, ] + [GATConv() for i in range(num_layers)]
#         self.time_encoder = None
#         self.edge_raw_embed = torch.nn.Embedding.from_pretrained(torch.from_numpy(n_feat.astype(np.float32)), freeze=True)
#         self.node_raw_embed = torch.nn.Embedding.from_pretrained(torch.from_numpy(n_feat.astype(np.float32)), freeze=True)



#     def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=20, edge_idx_preserve_list=None):
#         assert(curr_layers >= 0)

#         device = self.n_feat_th.device

#         # cut_time_l ???
#         src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
#         cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
#         src_node_feat = self.node_raw_embed(src_node_batch_th)


#         if curr_layers == 0:
#             return src_node_feat
        

#         # else:
#         src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor( 
#                                                                     src_idx_l, 
#                                                                     cut_time_l, 
#                                                                     num_neighbors=num_neighbors,
#                                                                     edge_idx_preserve_list=edge_idx_preserve_list)
        


#         edge_index = None
#         X = None
#         X = self.layers[curr_layers](X, edge_index)


#         # recursively,  use GATConv here

        
#         pass