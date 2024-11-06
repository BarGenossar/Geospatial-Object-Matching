import config
from utils import *
import faiss
import numpy as np
from collections import defaultdict


class Blocker:
    def __init__(self, object_dict):
        self.nn_param = config.Blocking.nn_param
        self.cand_pairs = config.Blocking.cand_pairs
        self.object_dict = object_dict
        self.centroids_dict = self._get_centroids()
        self.cands_mapping = {ind: orig_ind for ind, orig_ind in enumerate(self.centroids_dict['cands'])}
        self.index_mapping = {ind: orig_ind for ind, orig_ind in enumerate(self.centroids_dict['index'])}
        self.nn_dict, self.dist_dict = self._run_nn_search()
        self.pos_pairs, self.neg_pairs = self._get_candidate_pairs()

    def _get_centroids(self):
        centroids_dict = defaultdict(dict)
        for objects_type, objects_dict in self.object_dict.items():
            if objects_type not in ['cands', 'index']:
                continue
            for file_ind, vertices in objects_dict.items():
                centroids_dict[objects_type][file_ind] = get_centroid(vertices)
            centroids_dict[objects_type] = dict(sorted(centroids_dict[objects_type].items()))
        return centroids_dict

    @staticmethod
    def _get_ind_mapping(centroids):
        return {ind: orig_ind for ind, orig_ind in enumerate(sorted(centroids))}

    def _run_nn_search(self):
        nn_dict, dists_dict = {}, {}
        cands_centroids_np = np.array(list(self.centroids_dict['cands'].values()), dtype=np.float32)
        index_centroids_np = np.array(list(self.centroids_dict['index'].values()), dtype=np.float32)
        index = faiss.IndexFlatL2(index_centroids_np.shape[1])
        index.add(index_centroids_np)
        for i, query_centroid in enumerate(cands_centroids_np):
            dists, neighbors = index.search(np.array([query_centroid]), self.nn_param)
            nn_dict[self.cands_mapping[i]] = [self.index_mapping[ind] for ind in neighbors.flatten()]
            dists_dict[self.cands_mapping[i]] = [dist for dist in dists.flatten()]
        return nn_dict, dists_dict

    @staticmethod
    def _get_start_ind4nn(nn_inds, cand_ind):
        # todo: It seems like "data/240823/em/" has small number of duplicates between subsequent objects
        if nn_inds[0] + 1 == cand_ind:
            return 1
        else:
            return 0

    def _get_candidate_pairs(self):
        # todo: Support more complex cases of candidate pairs creation such as conditions on the distance of the
        #  negative pairs
        pos_pairs, neg_pairs = [], []
        local_mapping_dict = self._get_local_mapping_dict()
        for cand_ind, nn_inds in self.nn_dict.items():
            start_ind = self._get_start_ind4nn(nn_inds, cand_ind)
            for nn_ind in nn_inds[start_ind:start_ind + self.cand_pairs]:
                if local_mapping_dict['cands'][cand_ind] == local_mapping_dict['index'][nn_ind]:
                    pos_pairs.append((cand_ind, nn_ind))
                else:
                    neg_pairs.append((cand_ind, nn_ind))
        return pos_pairs, neg_pairs

    def _get_local_mapping_dict(self):
        """
        This function returns the mapping dictionary of the objects in the current object_dict.
        The mapping is required because in some datasets object file names are recognized as integers and in some
        datasets as uids
        """
        local_mapping_dict = defaultdict(dict)
        if 'mapping_dict' not in self.object_dict.keys():
            local_mapping_dict['cands'] = {ind: ind for ind in self.object_dict['cands'].keys()}
            local_mapping_dict['index'] = {ind: ind for ind in self.object_dict['index'].keys()}
        else:
            local_mapping_dict['cands'] = self.object_dict['mapping_dict']['cands']
            local_mapping_dict['index'] = self.object_dict['mapping_dict']['index']
        return local_mapping_dict

