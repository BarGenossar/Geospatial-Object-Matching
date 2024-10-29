from scipy.spatial import ConvexHull
import numpy as np
import faiss
import config
from shapely import Polygon, LineString
from scipy.spatial import KDTree
from collections import defaultdict
from utils import get_feature_name_list


class PairProcessor:
    def __init__(self, obj_att_vals, pairs_list):
        self.max_ratio_val = config.Constants.max_ratio_val
        self.obj_att_vals = obj_att_vals
        self.pairs_list = pairs_list
        self.feature_name_list = get_feature_name_list()
        self.feature_funcs_dict = self._get_feature_dict()
        self.feature_to_att_dict = self._get_feature_to_att_dict()
        self.feature_vec = self._create_feature_vectors()

    def _get_feature_dict(self):
        # TODO: Add more feature types except for ratio
        feature_dict = {feature: self._get_ratio for feature in self.feature_name_list if 'ratio' in feature}
        return feature_dict

    def _get_feature_to_att_dict(self):
         # TODO: Add more feature types except for ratio
        feature_to_att_dict = {feature: feature.split('_ratio')[0] for feature in
                               self.feature_name_list if 'ratio' in feature}
        return feature_to_att_dict

    def _get_feature_vec(self, pair):
        # The convention is that the first object in the pair is the candidate and the second belongs to the index
        feature_vec = []
        for feature_name in self.feature_name_list:
            try:
                feature_vec.append(min(self.max_ratio_val, self.feature_funcs_dict[feature_name](feature_name, pair)))
            except:
                print(f'Error in feature {feature_name} for pair {pair}')
                # feature_vec.append(np.nan)
                feature_vec.append(0.0)
        return feature_vec

    def _create_feature_vectors(self):
        feature_vectors = []
        for pair in self.pairs_list:
            feature_vectors.append(self._get_feature_vec(pair))
        return feature_vectors

    def _get_ratio(self, feature_name, pair):
        att_name = self.feature_to_att_dict[feature_name]
        cand_att_val = self.obj_att_vals[att_name]['cands'][pair[0]]
        index_att_val = self.obj_att_vals[att_name]['index'][pair[1]]
        return round(cand_att_val / index_att_val, 3)

