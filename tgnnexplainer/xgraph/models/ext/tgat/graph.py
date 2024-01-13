import numpy as np

class NeighborFinder:
    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """ 
        self.adj_list = adj_list
        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        
        self.off_set_l = off_set_l
        
        self.uniform = uniform

    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]
        
        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0,]
        
        for i in range(len(adj_list)): # adj_list: [[], [(...), ...], ...]
            curr = adj_list[i]
            # curr = sorted(curr, key=lambda x: x[1])
            curr = sorted(curr, key=lambda x: x[2]) # sort according to time
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            n_ts_l.extend([x[2] for x in curr])
           
            
            off_set_l.append(len(n_idx_l)) # the end index of this node's temporal interactions
            # import ipdb; ipdb.set_trace()
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert(len(n_idx_l) == len(n_ts_l))
        assert(off_set_l[-1] == len(n_ts_l))
        
        return n_idx_l, n_ts_l, e_idx_l, off_set_l
        
    def find_before(self, src_idx, cut_time):
        """
    
        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        
        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]] # one node's neighbors
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]

        # import ipdb; ipdb.set_trace()
        
        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_ts, neighbors_e_idx

        # left = 0
        # right = len(neighbors_idx) - 1

        
        # while left + 1 < right: # ! binary search, include cut_time
        #     mid = (left + right) // 2
        #     curr_t = neighbors_ts[mid]
        #     if curr_t <= cut_time:
        #         left = mid
        #     else:
        #         right = mid
            
        # if neighbors_ts[right] <= cut_time:
        #     end_point = right + 1
        # elif neighbors_ts[left] <= cut_time:
        #     end_point = left + 1
        # else:
        #     end_point = left

        
        # indices = neighbors_ts <= cut_time
        indices = neighbors_ts < cut_time # NOTE: important?

        # import ipdb; ipdb.set_trace()

        
        # return neighbors_idx[:end_point], neighbors_e_idx[:end_point], neighbors_ts[:end_point]
        # return neighbors_idx[:end_point], neighbors_e_idx[:end_point], neighbors_ts[:end_point]
        return neighbors_idx[indices], neighbors_e_idx[indices], neighbors_ts[indices]

        # if neighbors_ts[right] < cut_time: # https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs/issues/8
        #     return neighbors_idx[:right], neighbors_e_idx[:right], neighbors_ts[:right]
        # else:
        #     return neighbors_idx[:left], neighbors_e_idx[:left], neighbors_ts[:left]

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20, edge_idx_preserve_list=None):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        """
        assert(len(src_idx_l) == len(cut_time_l))
        
        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        
        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)
            # import ipdb; ipdb.set_trace()
            
            # if i == 1:
            #     import ipdb; ipdb.set_trace()

            if len(ngh_idx) > 0: #! only found neighbors list is not empty, otherwise all zeros
                if self.uniform:
                    raise NotImplementedError('Should not use this scheme')
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors) # sample 'num_neighbors' neighbors in the ngh_idx.
                    
                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
                    
                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                else: # we can use this setting to restrict the number of previous events observed.

                    # ngh_ts = ngh_ts[:num_neighbors]
                    # ngh_idx = ngh_idx[:num_neighbors]
                    # ngh_eidx = ngh_eidx[:num_neighbors]
                    

                    # get recent temporal edges
                    ngh_ts = ngh_ts[-num_neighbors:]
                    ngh_idx = ngh_idx[-num_neighbors:]
                    ngh_eidx = ngh_eidx[-num_neighbors:]

                    # mask out discarded edge_idxs, these should not be seen.
                    if edge_idx_preserve_list is not None:
                        mask = np.isin(ngh_eidx, edge_idx_preserve_list)
                        ngh_ts = ngh_ts[mask]
                        ngh_idx = ngh_idx[mask]
                        ngh_eidx = ngh_eidx[mask]
                    
                    assert(len(ngh_idx) <= num_neighbors)
                    assert(len(ngh_ts) <= num_neighbors)
                    assert(len(ngh_eidx) <= num_neighbors)
                    
                    # out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    # out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    # out_ngh_eidx_batch[i,  num_neighbors - len(ngh_eidx):] = ngh_eidx

                    # end positions already have been 0.
                    out_ngh_node_batch[i, :len(ngh_idx)] = ngh_idx
                    out_ngh_t_batch[i, :len(ngh_ts)] = ngh_ts
                    out_ngh_eidx_batch[i, :len(ngh_eidx)] = ngh_eidx
                    
        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    

            

