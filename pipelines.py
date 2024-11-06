import json
import os
import config
from utils import *
from blocking import Blocker
from collections import defaultdict
from process_pairs import PairProcessor
from object_properties import ObjectPropertiesProcessor
import numpy as np
from sklearn.model_selection import train_test_split
from classifier import FlexibleClassifier
from abc import ABC, abstractmethod


class PipelineManager:
    def __init__(self, seed, logger):
        self.dataset_name = config.Constants.dataset_name
        self.seed = seed
        self.logger = logger
        self.object_dict = None
        self.pos_pairs, self.neg_pairs = None, None
        self.neg_indices_train, self.neg_indices_test = None, None
        self.pos_indices_train, self.pos_indices_test = None, None
        self.split_data()
        self.dataset_dict = self._create_dataset_dict()
        self._train_and_evaluate()

    def _read_objects(self):
        # todo: Do we need to read more properties of the objects besides the vertices?
        dataset_config = json.load(open('dataset_configs.json'))[self.dataset_name]
        object_dict = getattr(self, f'_read_objects_{self.dataset_name}')(dataset_config)
        return object_dict

    @staticmethod
    def _read_object_path_dict(dataset_config):
        return {'cands': dataset_config['cands_path'], 'index': dataset_config['index_path']}

    def _read_objects_bo_em(self, dataset_config):
        objects_path_dict = self._read_object_path_dict(dataset_config)
        object_dict = defaultdict(dict)
        for objects_type, objects_path in objects_path_dict.items():
            for filename in os.listdir(objects_path):
                file_ind = int(filename.split('.')[0])
                json_data = read_json(objects_path, file_ind)
                vertices = json_data['vertices']
                object_dict[objects_type][file_ind] = close_polygon(vertices)
            object_dict[objects_type] = dict(sorted(object_dict[objects_type].items()))
        return object_dict

    def _read_objects_gpkg(self, dataset_config):
        objects_path_dict = self._read_object_path_dict(dataset_config)
        object_dict = defaultdict(dict)
        for objects_type, objects_path in objects_path_dict.items():
            for filename in os.listdir(objects_path):
                file_ind = int(filename.split('.')[0])
                json_data = read_json(objects_path, file_ind)
                json_data = json.loads(json_data)
                vertices = json_data['vertices']
                object_dict[objects_type][file_ind] = close_polygon(vertices)
            object_dict[objects_type] = dict(sorted(object_dict[objects_type].items()))
        return object_dict

    def _read_objects_delivery3(self, dataset_config):
        objects_path_dict = self._read_object_path_dict(dataset_config)
        object_dict = defaultdict(dict)
        mapping_dict = defaultdict(dict)
        for objects_type, objects_path in objects_path_dict.items():
            for file_ind, filename in enumerate(os.listdir(objects_path)):
                filename = int(filename.split('.')[0])
                json_data = read_json(objects_path, file_ind)
                vertices = json_data['vertices']
                object_dict[objects_type][file_ind] = close_polygon(vertices)
                mapping_dict[objects_type][file_ind] = filename
            object_dict[objects_type] = dict(sorted(object_dict[objects_type].items()))
        object_dict['mapping_dict'] = mapping_dict
        return object_dict

    def split_data(self):
        if config.Constants.load_dataset_dict:
            return
        self.object_dict = self._read_objects()
        self.pos_pairs, self.neg_pairs = self._run_blocker()
        self.neg_indices_train, self.neg_indices_test = self._split_indices(0)
        self.pos_indices_train, self.pos_indices_test = self._split_indices(1)
        return

    def _merge_labels(self):
        neg_labels_train = [0 for _ in self.neg_indices_train]
        neg_labels_test = [0 for _ in self.neg_indices_test]
        pos_labels_train = [1 for _ in self.pos_indices_train]
        pos_labels_test = [1 for _ in self.pos_indices_test]
        merged_train = neg_labels_train + pos_labels_train
        merged_test = neg_labels_test + pos_labels_test
        return merged_train, merged_test

    def _run_blocker(self):
        blocker = Blocker(self.object_dict)
        return blocker.pos_pairs, blocker.neg_pairs

    def _split_indices(self, label):
        test_size = config.Constants.test_ratio
        pairs = self.neg_pairs if label == 0 else self.pos_pairs
        indices = list(range(len(pairs)))
        indices_train, indices_test = train_test_split(indices, test_size=test_size, random_state=label+self.seed)
        return set(indices_train), set(indices_test)

    @abstractmethod
    def _create_dataset_dict(self):
        pass

    @abstractmethod
    def _train_and_evaluate(self):
        pass


class PipelineManagerClassicModels(PipelineManager):
    def __init__(self, seed, logger):
        super().__init__(seed, logger)

    def _create_dataset_dict(self):
        if config.Constants.load_dataset_dict:
            return load_dataset_dict(self.logger, self.seed)
        feature_dict = self._generate_feature_dict()
        dataset_dict = self._create_final_dict(feature_dict)
        if config.Constants.save_dataset_dict:
            save_dataset_dict(dataset_dict, self.seed, self.logger)
        return dataset_dict

    def _generate_feature_dict(self):
        feature_dict = defaultdict(dict)
        obj_prop_vals = ObjectPropertiesProcessor(self.object_dict).prop_vals_dict
        for label, pairs_list in zip([0, 1], [self.neg_pairs, self.pos_pairs]):
            feature_dict[label] = PairProcessor(obj_prop_vals, pairs_list).feature_vec
        return feature_dict

    def _create_final_dict(self, feature_dict):
        dataset_dict = defaultdict(dict)
        np.random.seed(self.seed)
        merged_features_train, merged_features_test = self._merge_features(feature_dict)
        merged_labels_train, merged_labels_test = self._merge_labels()
        dataset_dict = self._prepare_dataset(dataset_dict, 'train', merged_features_train, merged_labels_train)
        dataset_dict = self._prepare_dataset(dataset_dict, 'test', merged_features_test, merged_labels_test)
        return dataset_dict

    @staticmethod
    def _prepare_dataset(dataset_dict, file_type, merged_features, merged_labels):
        combined = list(zip(merged_features, merged_labels))
        np.random.shuffle(combined)
        dataset_dict[file_type]['X'] = np.array([elem[0] for elem in combined])
        dataset_dict[file_type]['Y'] = np.array([elem[1] for elem in combined])
        return dataset_dict

    def _merge_features(self, feature_dict):
        neg_pairs_train = [feature_dict[0][ind] for ind in self.neg_indices_train]
        neg_pairs_test = [feature_dict[0][ind] for ind in self.neg_indices_test]
        pos_pairs_train = [feature_dict[1][ind] for ind in self.pos_indices_train]
        pos_pairs_test = [feature_dict[1][ind] for ind in self.pos_indices_test]
        merged_train = neg_pairs_train + pos_pairs_train
        merged_test = neg_pairs_test + pos_pairs_test
        return merged_train, merged_test

    def _train_and_evaluate(self):
        params_dict = self._read_config_models()
        load_trained_models = config.Models.load_trained_models
        cv = config.Models.cv
        FlexibleClassifier(self.dataset_dict, params_dict, self.seed, self.logger, load_trained_models, cv)
        return

    @staticmethod
    def _read_config_models():
        model_list = config.Models.model_list
        params_dict = dict()
        for model in model_list:
            params_dict[model] = config.Models.params_dict[model]
        return params_dict



class PipelineManagerGNN(PipelineManager):
    def __init__(self, seed, logger):
        super().__init__(seed, logger)
        self.dataset_dict = self._create_dataset_dict()
        self._train_and_evaluate()

    def _create_dataset_dict(self):
        if config.Constants.load_dataset_dict:
            return load_dataset_dict(self.logger, self.seed)
        # todo: change
        feature_dict = self._generate_feature_dict()
        dataset_dict = self._create_final_dict(feature_dict)
        if config.Constants.save_dataset_dict:
            save_dataset_dict(dataset_dict, self.seed, self.logger)
        return dataset_dict

    def _generate_feature_dict(self):
        feature_dict = defaultdict(dict)
        obj_prop_vals = ObjectPropertiesProcessor(self.object_dict).prop_vals_dict
        for label, pairs_list in zip([0, 1], [self.neg_pairs, self.pos_pairs]):
            feature_dict[label] = PairProcessor(obj_prop_vals, pairs_list).feature_vec
        return feature_dict



