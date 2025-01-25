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
    def __init__(self, seed, logger, min_surfaces_num=10):
        self.dataset_name = config.Constants.dataset_name
        self.seed = seed
        self.logger = logger
        self.min_surfaces_num = min_surfaces_num
        self.object_dict = None
        self.pos_pairs, self.neg_pairs = None, None
        self.neg_indices_train, self.neg_indices_test = None, None
        self.pos_indices_train, self.pos_indices_test = None, None
        self.dataset_dict = self._create_dataset_dict()
        self.result_dict = self._train_and_evaluate()

    def _read_objects(self):
        # todo: Do we need to read more properties of the objects besides the vertices?
        dataset_config = json.load(open('dataset_configs.json'))[self.dataset_name]
        if config.Constants.load_object_dict:
            object_dict = load_object_dict(self.logger, dataset_config['object_dict_path'])
            if object_dict is not None:
                return object_dict
        object_dict = getattr(self, f'_read_objects_{self.dataset_name}')(dataset_config)
        if config.Constants.save_object_dict:
            print("Saving object_dict")
            joblib.dump(object_dict, dataset_config['object_dict_path'])
        return object_dict

    @staticmethod
    def _compute_object_centroid(polygon_mesh):
        vertices = [tuple(coord) for surface in polygon_mesh for coord in surface]
        unique_vertices = np.array(list(set(vertices)))
        return unique_vertices.mean(axis=0)

    @staticmethod
    def _standardize_obj_key(obj_key, object_type):
        if object_type == 'cands':
            return obj_key.split('bag_')[1]
        elif object_type == 'index':
            return obj_key.split('NL.IMBAG.Pand.')[1].split('-0')[0]
        else:
            raise ValueError('Invalid source')

    def _get_polygon_mesh(self, obj_data, obj_key, vertices):
        boundaries = obj_data['CityObjects'][obj_key]['geometry'][0]['boundaries'][0]
        if len(boundaries) < self.min_surfaces_num:
            return None
        polygon_mesh = []
        for surface in boundaries:
            polygon_mesh.append([vertices[i] for sub_surface_list in surface for i in sub_surface_list])
        centroid = self._compute_object_centroid(polygon_mesh)
        return {'polygon_mesh': polygon_mesh, 'centroid': centroid}

    def _insert_polygon_mesh(self, object_dict, obj_type, obj_data, obj_ind=None):
        vertices = obj_data['vertices']
        obj_key = list(obj_data['CityObjects'].keys())[0]
        polygon_mesh = self._get_polygon_mesh(obj_data, obj_key, vertices)
        if polygon_mesh is not None:
            object_dict[obj_type][obj_ind] = polygon_mesh
        return object_dict

    def _read_objects_bo_em(self, dataset_config):
        objects_path_dict = read_object_path_dict(dataset_config)
        object_dict = defaultdict(dict)
        for objects_type, objects_path in objects_path_dict.items():
            for filename in os.listdir(objects_path):
                file_ind = int(filename.split('.')[0])
                json_data = read_json(objects_path, file_ind)
                object_dict = self._insert_polygon_mesh(object_dict, objects_type, json_data, file_ind)
            object_dict[objects_type] = dict(sorted(object_dict[objects_type].items()))
        return object_dict

    def _read_objects_gpkg(self, dataset_config):
        objects_path_dict = read_object_path_dict(dataset_config)
        object_dict = defaultdict(dict)
        for objects_type, objects_path in objects_path_dict.items():
            for filename in os.listdir(objects_path):
                file_ind = int(filename.split('.')[0])
                json_data = read_json(objects_path, file_ind)
                json_data = json.loads(json_data)
                object_dict = self._insert_polygon_mesh(object_dict, objects_type, json_data, file_ind)
            object_dict[objects_type] = dict(sorted(object_dict[objects_type].items()))
        return object_dict

    def _read_objects_delivery3(self, dataset_config):
        objects_path_dict = read_object_path_dict(dataset_config)
        object_dict = defaultdict(dict)
        mapping_dict = defaultdict(dict)
        inv_mapping_dict = defaultdict(dict)
        for objects_type, objects_path in objects_path_dict.items():
            for file_ind, file_name in enumerate(os.listdir(objects_path)):
                file_name = file_name.split('.')[0]
                json_data = read_json(objects_path, file_name)
                object_dict = self._insert_polygon_mesh(object_dict, objects_type, json_data, file_ind)
                mapping_dict[objects_type][file_ind] = file_name
                inv_mapping_dict[objects_type][file_name] = file_ind
            object_dict[objects_type] = dict(sorted(object_dict[objects_type].items()))
        object_dict['mapping_dict'] = mapping_dict
        object_dict['inv_mapping_dict'] = inv_mapping_dict
        return object_dict

    def _read_objects_Hague(self, dataset_config):
        objects_path_dict = read_object_path_dict(dataset_config)
        object_dict = defaultdict(dict)
        for objects_type, objects_path in objects_path_dict.items():
            print(f"Reading {objects_type} objects")
            file_list = [f for f in os.listdir(objects_path) if f.endswith('.json')]
            for file_ind, file_name in enumerate(file_list):
                print(f"File number {file_ind + 1} out of {len(file_list)}")
                file_path = ''.join([objects_path, file_name])
                with open(file_path, 'r') as f:
                    data = json.load(f)
                vertices = data['vertices']
                for obj_key in data['CityObjects'].keys():
                    try:
                        new_obj_key = self._standardize_obj_key(obj_key, objects_type)
                        object_dict[objects_type][new_obj_key] = self._get_polygon_mesh(data, obj_key, vertices)
                    except:
                        continue
            object_dict['mapping_dict'][objects_type] = {ind: obj_key for ind, obj_key in
                                                         enumerate(object_dict[objects_type].keys())}
            object_dict['inv_mapping_dict'][objects_type] = {obj_key: ind for ind, obj_key in
                                                             enumerate(object_dict[objects_type].keys())}
        return object_dict

    @staticmethod
    def read_objects_synthetic(self, dataset_config):
        pass

    def split_data(self):
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
        blocking_method = config.Blocking.blocking_method
        self.logger.info(f"Running blocking method {blocking_method}")
        blocker = Blocker(self.object_dict)
        self.logger.info(f"The blocking process ended successfully")
        return blocker.pos_pairs, blocker.neg_pairs

    def _split_indices(self, label):
        test_size = config.Constants.test_ratio
        pairs = self.neg_pairs if label == 0 else self.pos_pairs
        indices = list(range(len(pairs)))
        indices_train, indices_test = train_test_split(indices, test_size=test_size, random_state=label+self.seed)
        return set(indices_train), set(indices_test)

    def _create_dataset_dict(self):
        if config.Constants.load_dataset_dict:
            dataset_dict = load_dataset_dict(self.logger, self.seed)
            if dataset_dict is not None:
                return dataset_dict
        self.split_data()
        feature_dict = self._generate_feature_dict()
        dataset_dict = self._create_final_dict(feature_dict)
        if config.Constants.save_dataset_dict:
            save_dataset_dict(dataset_dict, self.seed, self.logger)
        return dataset_dict

    def _generate_feature_dict(self):
        feature_dict = defaultdict(dict)
        obj_prop_vals = ObjectPropertiesProcessor(self.object_dict).prop_vals_dict
        if config.Constants.save_properties_dict:
            save_properties_dict(obj_prop_vals, self.seed, self.logger)
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

    def _merge_features(self, feature_dict):
        neg_pairs_train = [feature_dict[0][ind] for ind in self.neg_indices_train]
        neg_pairs_test = [feature_dict[0][ind] for ind in self.neg_indices_test]
        pos_pairs_train = [feature_dict[1][ind] for ind in self.pos_indices_train]
        pos_pairs_test = [feature_dict[1][ind] for ind in self.pos_indices_test]
        merged_train = neg_pairs_train + pos_pairs_train
        merged_test = neg_pairs_test + pos_pairs_test
        return merged_train, merged_test

    @staticmethod
    def _prepare_dataset(dataset_dict, file_type, merged_features, merged_labels):
        combined = list(zip(merged_features, merged_labels))
        np.random.shuffle(combined)
        dataset_dict[file_type]['X'] = np.array([elem[0] for elem in combined])
        dataset_dict[file_type]['Y'] = np.array([elem[1] for elem in combined])
        return dataset_dict

    def _train_and_evaluate(self):
        params_dict = self._read_config_models()
        load_trained_models = config.Models.load_trained_models
        cv = config.Models.cv
        flexible_classifier = FlexibleClassifier(self.dataset_dict, params_dict, self.seed,
                                                 self.logger, load_trained_models, cv)
        return flexible_classifier.result_dict

    @staticmethod
    def _read_config_models():
        model_list = config.Models.model_list
        params_dict = dict()
        for model in model_list:
            params_dict[model] = config.Models.params_dict[model]
        return params_dict
