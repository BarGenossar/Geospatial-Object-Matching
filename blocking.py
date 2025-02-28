import config
from utils import *
import faiss
import numpy as np
from collections import defaultdict
from time import time
from scipy.spatial import KDTree


class Blocker:
    def __init__(self, object_dict, property_dict, feature_importance_scores, property_ratios,
                 blocking_method, train_or_test):
        self.blocking_method = blocking_method
        self.nn_param = config.Blocking.nn_param
        self.cand_pairs_per_item_list = config.Blocking.cand_pairs_per_item_list
        self.bkafi_dim_list = config.Blocking.bkafi_dim_list
        self.nbits = config.Blocking.nbits
        self.object_dict = object_dict
        self.property_dict = property_dict
        self.feature_importance_scores = feature_importance_scores
        self.property_ratios = property_ratios
        self.train_or_test = train_or_test
        self.centroids_dict = self._get_centroids()
        self.cands_mapping = {ind: orig_ind for ind, orig_ind in enumerate(self.object_dict['cands'].keys())}
        self.index_mapping = {ind: orig_ind for ind, orig_ind in enumerate(self.object_dict['index'].keys())}
        self.blocking_method_dict = self._get_blocking_method_dict()
        self.nn_dict, self.dist_dict = self._run_blocking()
        self.pos_pairs_dict, self.neg_pairs_dict = self._get_candidate_pairs()

    def _get_centroids(self):
        centroids_dict = defaultdict(dict)
        for objects_type in ['cands', 'index']:
            centroids_dict[objects_type] = {obj_ind: obj_data['centroid'] for obj_ind, obj_data in
                                           self.object_dict[objects_type].items()}
            centroids_dict[objects_type] = dict(sorted(centroids_dict[objects_type].items()))
        return centroids_dict

    @staticmethod
    def _get_ind_mapping(centroids):
        return {ind: orig_ind for ind, orig_ind in enumerate(sorted(centroids))}

    def _get_blocking_method_dict(self):
        blocking_method_dict = {'exhaustive': self._run_exhaustive,
                                'lsh': self._run_lsh,
                                'kdtree': self._run_kdtree,
                                'bkafi': self._run_bkafi}
        return blocking_method_dict

    def _run_blocking(self):
        nn_dict, dists_dict = self.blocking_method_dict[self.blocking_method]()
        return nn_dict, dists_dict


    def _run_exhaustive(self):
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

    def _run_lsh(self):
        nn_dict, dists_dict = {}, {}
        cands_centroids_np = np.array(list(self.centroids_dict['cands'].values()), dtype=np.float32)
        index_centroids_np = np.array(list(self.centroids_dict['index'].values()), dtype=np.float32)
        index = faiss.IndexLSH(index_centroids_np.shape[1], self.nbits)
        index.add(index_centroids_np)
        for i, query_centroid in enumerate(cands_centroids_np):
            dists, neighbors = index.search(np.array([query_centroid]), self.nn_param)
            nn_dict[self.cands_mapping[i]] = [self.index_mapping[ind] for ind in neighbors.flatten()]
            dists_dict[self.cands_mapping[i]] = [dist for dist in dists.flatten()]
        return nn_dict, dists_dict

    def _run_kdtree(self, search_dict):
        nn_dict, dists_dict = {}, {}
        cands_vectors_np = np.array(list(search_dict['cands'].values()), dtype=np.float32)
        index_vectors_np = np.array(list(search_dict['index'].values()), dtype=np.float32)
        index = KDTree(index_vectors_np)
        dists, neighbors = index.query(cands_vectors_np, self.nn_param)
        for i, query_centroid in enumerate(cands_vectors_np):
            nn_dict[self.cands_mapping[i]] = [self.index_mapping[ind] for ind in neighbors[i]]
            dists_dict[self.cands_mapping[i]] = [round(dist, 3) for dist in dists[i]]
        return nn_dict, dists_dict

    def _run_bkafi(self):
        if self.train_or_test == 'train':
            return self._run_bkafi_train()
        nn_dict, dists_dict = {}, {}
        model_name = config.Models.blocking_model
        for bkafi_dim in self.bkafi_dim_list:
            target_blocking_features = {feature: self.property_ratios[feature.split('_ratio')[0]]
                                        for feature, _ in self.feature_importance_scores[model_name][:bkafi_dim]}
            bkafi_dict = self._get_bkafi_dict(target_blocking_features)
            nn_dict[bkafi_dim], dists_dict[bkafi_dim] = self._run_kdtree(bkafi_dict)
        return nn_dict, dists_dict

    def _run_bkafi_train(self):
        nn_dict, dists_dict = {}, {}
        model_name = config.Models.blocking_model
        bkafi_dim = len(self.feature_importance_scores[model_name])
        target_blocking_features = {feature: self.property_ratios[feature.split('_ratio')[0]]
                                    for feature, _ in self.feature_importance_scores[model_name][:bkafi_dim]}
        bkafi_dict = self._get_bkafi_dict(target_blocking_features)
        nn_dict[bkafi_dim], dists_dict[bkafi_dim] = self._run_kdtree(bkafi_dict)
        return nn_dict, dists_dict

    def _get_bkafi_dict(self, target_blocking_features):
        bkafi_dict = defaultdict(dict)
        factor_dict = self._get_bkafi_factor_dict(target_blocking_features)
        for obj_type in ['cands', 'index']:
            for obj_ind in self.object_dict[obj_type].keys():
                bkafi_dict[obj_type][obj_ind] = []
                for feature in target_blocking_features:
                    property_val = self.property_dict[feature.split('_ratio')[0]][obj_type][obj_ind]
                    bkafi_dict[obj_type][obj_ind].append(property_val * factor_dict[obj_type][feature])
                bkafi_dict[obj_type][obj_ind] = np.array(bkafi_dict[obj_type][obj_ind])
        return bkafi_dict

    @staticmethod
    def _get_bkafi_factor_dict(target_blocking_features):
        factor_dict = defaultdict(dict)
        factor_dict['cands'] = {feature: target_blocking_features[feature]['mean']
                                for feature in target_blocking_features}
        factor_dict['index'] = {feature: 1.0 for feature in target_blocking_features}
        return factor_dict

    @staticmethod
    def _get_start_ind4nn(nn_inds, cand_ind):
        if nn_inds[0] + 1 == cand_ind:
            return 1
        else:
            return 0

    def _get_candidate_pairs(self):
        # todo: Support more complex cases of candidate pairs creation such as conditions on the distance of the
        #  negative pairs
        if self.train_or_test == 'train':
            cand_pairs_per_item_list = [self.cand_pairs_per_item_list[0]]
        else:
            cand_pairs_per_item_list = self.cand_pairs_per_item_list
        pos_pairs_dict, neg_pairs_dict = defaultdict(dict), defaultdict(dict)
        # local_mapping_dict = self._get_local_mapping_dict()
        for bkafi_dim in self.nn_dict.keys():
            for list_ind, cand_pairs_per_item in enumerate(cand_pairs_per_item_list):
                if list_ind == 0:
                    pos_pairs_dict[bkafi_dim][cand_pairs_per_item] = []
                    neg_pairs_dict[bkafi_dim][cand_pairs_per_item] = []
                else:
                    previous_val = self.cand_pairs_per_item_list[list_ind - 1]
                    pos_pairs_dict[bkafi_dim][cand_pairs_per_item] = pos_pairs_dict[bkafi_dim][previous_val].copy()
                    neg_pairs_dict[bkafi_dim][cand_pairs_per_item] = neg_pairs_dict[bkafi_dim][previous_val].copy()
                for cand_ind, nn_inds in self.nn_dict[bkafi_dim].items():
                    start_ind = self.cand_pairs_per_item_list[list_ind - 1] if list_ind > 0 else 0
                    cand_ind = str(cand_ind)
                    for nn_ind in nn_inds[start_ind:cand_pairs_per_item]:
                        if cand_ind == nn_ind:
                            pos_pairs_dict[bkafi_dim][cand_pairs_per_item].append((cand_ind, nn_ind))
                        else:
                            neg_pairs_dict[bkafi_dim][cand_pairs_per_item].append((cand_ind, nn_ind))
                        # if local_mapping_dict['cands'][cand_ind] == local_mapping_dict['index'][nn_ind]:
                        #     pos_pairs_dict[bkafi_dim][cand_pairs_per_item].append((cand_ind, nn_ind))
                        # else:
                        #     neg_pairs_dict[bkafi_dim][cand_pairs_per_item].append((cand_ind, nn_ind))
        return pos_pairs_dict, neg_pairs_dict

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

