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
    def __init__(self, seed, logger, with_prep_training=False, min_surfaces_num=10):
        self.dataset_name = config.Constants.dataset_name
        self.seed = seed
        self.logger = logger
        self.with_prep_training = with_prep_training
        self.min_surfaces_num = min_surfaces_num
        self.evaluation_mode = config.Constants.evaluation_mode
        self.object_dict = None
        self.pos_pairs, self.neg_pairs = None, None
        self.neg_indices_train, self.neg_indices_test = None, None
        self.pos_indices_train, self.pos_indices_test = None, None
        self.prep_feature_importance_scores, self.prep_property_ratios = None, None
        self.dataset_dict = self._create_dataset_dict()
        self.flexible_classifier_obj = self._train_and_evaluate(self.dataset_dict, prep_mode=False)

    def _read_objects(self):
        dataset_config = json.load(open('dataset_configs.json'))[self.dataset_name]
        object_dict_path = dataset_config['object_dict_path']
        prep_object_dict_path = object_dict_path.replace('object_dict', 'prep_object_dict')
        if config.Constants.load_object_dict:
            object_dict = load_object_dict(self.logger, object_dict_path, 'object_dict')
            prep_object_dict = load_object_dict(self.logger, prep_object_dict_path, 'prep_object_dict')
            if object_dict is not None and prep_object_dict is not None:
                return object_dict, prep_object_dict
        self.logger.info("Generating object dict and preparatory object dict")
        object_dict = getattr(self, f'_read_objects_{self.dataset_name}')(dataset_config)
        object_dict, prep_object_dict = self._get_prep_training_object_ids(object_dict)
        if config.Constants.save_object_dict:
            self._save_object_dicts(object_dict, prep_object_dict, dataset_config)
        return object_dict, prep_object_dict

    def _save_object_dicts(self, object_dict, prep_object_dict, dataset_config):
        self.logger.info(f"Saving object dict to {dataset_config['object_dict_path']}")
        joblib.dump(object_dict, dataset_config['object_dict_path'])
        self.logger.info(f"Saving preparatory training object dict to {dataset_config['object_dict_path']}")
        prep_object_dict_path = dataset_config['object_dict_path'].replace('object_dict', 'prep_object_dict')
        joblib.dump(prep_object_dict, prep_object_dict_path)
        return

    def _get_prep_training_object_ids(self, object_dict):
        """
        Get the object ids for the preparatory training phase from cands by randomly selecting the ids from the
        object_dict['cands'].keys(). Then take them from the object_dict['index'].

        Parameters:
            object_dict (dict): The object dictionary containing the cands and index objects.

        Returns:

        """
        np.random.seed(self.seed)
        cands_ids = list(object_dict['cands'].keys())
        index_ids = set(object_dict['index'].keys())
        pos_samples_num = config.PreparatoryPhase.pos_pairs_num
        intersection_list = list(set(cands_ids).intersection(index_ids))
        prep_ids = set(np.random.choice(intersection_list, pos_samples_num, replace=False))
        prep_object_dict = {'cands': {object_id: object_dict['cands'][object_id] for object_id in prep_ids},
                            'index': {object_id: object_dict['index'][object_id] for object_id in prep_ids}}
        object_dict = self._remove_prep_objects_from_object_dict(object_dict, prep_ids)
        return object_dict, prep_object_dict

    @staticmethod
    def _remove_prep_objects_from_object_dict(object_dict, prep_ids):
        for objects_type in object_dict.keys():
            object_dict[objects_type] = {
                object_id: object_data
                for object_id, object_data in object_dict[objects_type].items()
                if object_id not in prep_ids
            }
        return object_dict


    @staticmethod
    def _compute_object_centroid(vertices):
        unique_vertices = np.array(vertices)
        return unique_vertices.mean(axis=0)

    @staticmethod
    def _get_vertices(polygon_mesh):
        return np.unique(np.array([coord for surface in polygon_mesh for coord in surface]), axis=0)

    def _get_polygon_mesh(self, obj_data, obj_key, vertices):
        boundaries = obj_data['CityObjects'][obj_key]['geometry'][0]['boundaries'][0]
        if len(boundaries) < self.min_surfaces_num:
            return None

        polygon_mesh = []
        for surface in boundaries:
            polygon_mesh.append([vertices[i] for sub_surface_list in surface for i in sub_surface_list])
        vertices = self._get_vertices(polygon_mesh)
        centroid = self._compute_object_centroid(vertices)
        return {'polygon_mesh': polygon_mesh, 'vertices': vertices, 'centroid': centroid}

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
                        new_obj_key = standardize_obj_key(obj_key, objects_type)
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

    def _split_data(self):
        self.neg_indices_train, self.neg_indices_test = self._split_indices(0)
        self.pos_indices_train, self.pos_indices_test = self._split_indices(1)
        return

    def _merge_labels(self):
        neg_label_train = [0] * len(self.neg_indices_train)
        neg_label_test = [0] * len(self.neg_indices_test)
        pos_label_train = [1] * len(self.pos_indices_train)
        pos_label_test = [1] * len(self.pos_indices_test)
        merged_train = neg_label_train + pos_label_train
        merged_test = neg_label_test + pos_label_test
        return merged_train, merged_test

    def _generate_prep_pairs(self):
        np.random.seed(self.seed)
        index_ids = list(self.prep_object_dict['index'].keys())
        pos_pairs = [(obj_id, obj_id) for obj_id in self.prep_object_dict['cands'].keys()]
        neg_pairs = [(obj_id, np.random.choice(index_ids)) for obj_id in self.prep_object_dict['cands'].keys()]
        neg_pairs = [(cand_id, index_id) for cand_id, index_id in neg_pairs if cand_id != index_id]
        return pos_pairs, neg_pairs

    def _run_blocker(self):
        blocking_method = config.Blocking.blocking_method
        self.logger.info(f"Running blocking method {blocking_method}")
        blocker = Blocker(self.object_dict, self.property_dict, self.prep_feature_importance_scores,
                          self.prep_property_ratios)
        self.logger.info(f"The blocking process ended successfully")
        pos_pairs, neg_pairs = blocker.pos_pairs, blocker.neg_pairs
        save_blocking_output(pos_pairs, neg_pairs, self.seed, self.logger)
        if self.evaluation_mode == 'blocking':
            self._evaluate_blocking(pos_pairs, neg_pairs)
        return blocker.pos_pairs, blocker.neg_pairs

    def _evaluate_blocking(self, pos_pairs, neg_pairs):
        index_ids = set(self.object_dict['index'].keys())
        cand_ids = set(self.object_dict['cands'].keys())
        max_intersection = index_ids.intersection(cand_ids)
        blocking_recall = round(len(pos_pairs) / len(max_intersection), 3)
        total_pairs = len(pos_pairs) + len(neg_pairs)
        self.logger.info(f"Blocking recall: {blocking_recall}")
        self.logger.info(f"Total pairs: {total_pairs}")
        save_blocking_evaluation(blocking_recall, total_pairs, self.seed, self.logger)
        return

    def _split_indices(self, label):
        test_size = config.Constants.test_ratio
        pairs = self.neg_pairs if label == 0 else self.pos_pairs
        indices = list(range(len(pairs)))
        indices_train, indices_test = train_test_split(indices, test_size=test_size, random_state=label+self.seed)
        return set(indices_train), set(indices_test)

    def _create_dataset_dict(self):
        dataset_dict = self._load_dataset_dict_wrapper()
        if dataset_dict is not None:
            return dataset_dict
        self.object_dict, self.prep_object_dict = self._read_objects()
        self.prep_pos_pairs, self.prep_neg_pairs = self._generate_prep_pairs()
        if self.with_prep_training:
            self.prep_feature_importance_scores, self.prep_property_ratios = self._run_prep_pipeline()
        self.property_dict = self._generate_property_dict()
        self.pos_pairs, self.neg_pairs = self._run_blocker()
        if self.evaluation_mode == "blocking":
            return
        self._split_data()
        feature_dict = self._generate_feature_dict()
        dataset_dict = self._create_final_dict(feature_dict)
        if config.Constants.save_dataset_dict:
            save_dataset_dict(dataset_dict, self.seed, self.logger)
        return dataset_dict

    def _load_dataset_dict_wrapper(self):
        dataset_dict = None
        if config.Constants.load_dataset_dict:
            dataset_dict = load_dataset_dict(self.logger, self.seed)
            if dataset_dict is not None:
                return dataset_dict
        return dataset_dict

    def _run_prep_pipeline(self):
        if config.Constants.load_prep_items:
            feature_importance_scores, matching_pairs_property_ratios = self._load_prep_items()
            if feature_importance_scores is not None and matching_pairs_property_ratios is not None:
                return feature_importance_scores, matching_pairs_property_ratios
        prep_feature_dict = self._generate_prep_feature_dict()
        prep_final_dict = self._create_prep_final_dict(prep_feature_dict)
        flexible_classifier_obj = self._train_and_evaluate(prep_final_dict, prep_mode=True)
        feature_importance_dict = flexible_classifier_obj.feature_importance_extraction()
        prep_property_ratios = flexible_classifier_obj.get_property_ratios(prep_mode=True)
        return feature_importance_dict, prep_property_ratios

    def _load_prep_items(self):
        feature_importance_dict, matching_pairs_property_ratios = None, None
        try:
            feature_importance_dict = load_feature_importance_scores(self.seed, self.logger)
            matching_pairs_property_ratios = load_property_ratios(self.seed, self.logger)
        except:
            self.logger.info("Could not load preparatory phase items. Running preparatory phase pipeline")
        return feature_importance_dict, matching_pairs_property_ratios

    def _generate_property_dict(self):
        if config.Constants.load_property_dict:
            property_dict = load_property_dict(self.seed, self.logger, prep_mode=False)
            if property_dict is not None:
                return property_dict
        self.logger.info("Generating property dictionary")
        obj_prop_vals = ObjectPropertiesProcessor(self.object_dict).prop_vals_dict
        if config.Constants.save_property_dict:
            save_property_dict(obj_prop_vals, self.seed, self.logger, prep_mode=False)
        return obj_prop_vals

    def _generate_feature_dict(self):
        feature_dict = defaultdict(dict)
        self.logger.info("Generating feature vectors")
        for label, pairs_list in zip([0, 1], [self.neg_pairs, self.pos_pairs]):
            feature_dict[label] = PairProcessor(self.property_dict, pairs_list).feature_vec
        return feature_dict

    def _generate_prep_feature_dict(self):
        prep_feature_dict = defaultdict(dict)
        self.logger.info("Generating preparatory phase property dictionary")
        prep_obj_prop_vals = ObjectPropertiesProcessor(self.prep_object_dict).prop_vals_dict
        if config.Constants.save_property_dict:
            save_property_dict(prep_obj_prop_vals, self.seed, self.logger, prep_mode=True)
        self.logger.info("Generating preparatory phase feature vectors")
        for label, pairs_list in zip([0, 1], [self.prep_neg_pairs, self.prep_pos_pairs]):
            prep_feature_dict[label] = PairProcessor(prep_obj_prop_vals, pairs_list).feature_vec
        return prep_feature_dict

    def _create_final_dict(self, feature_dict):
        dataset_dict = defaultdict(dict)
        np.random.seed(self.seed)
        merged_features_train, merged_features_test = self._merge_features(feature_dict)
        merged_labels_train, merged_labels_test = self._merge_labels()
        dataset_dict = self._prepare_dataset(dataset_dict, 'train', merged_features_train, merged_labels_train)
        dataset_dict = self._prepare_dataset(dataset_dict, 'test', merged_features_test, merged_labels_test)
        return dataset_dict

    def _create_prep_final_dict(self, feature_dict):
        prep_dataset_dict = defaultdict(dict)
        np.random.seed(self.seed)
        merged_features = feature_dict[0] + feature_dict[1]
        labels = [0] * len(self.prep_neg_pairs) + [1] * len(self.prep_pos_pairs)
        prep_dataset_dict = self._prepare_dataset(prep_dataset_dict, 'prep', merged_features, labels)
        if config.Constants.save_dataset_dict:
            save_dataset_dict(prep_dataset_dict, self.seed, self.logger)
        return prep_dataset_dict

    def _merge_features(self, feature_dict):
        neg_pairs_train = [feature_dict[0][ind] for ind in self.neg_indices_train]
        neg_pairs_test = [feature_dict[0][ind] for ind in self.neg_indices_test]
        pos_pairs_train = [feature_dict[1][ind] for ind in self.pos_indices_train]
        pos_pairs_test = [feature_dict[1][ind] for ind in self.pos_indices_test]
        merged_train = neg_pairs_train + pos_pairs_train
        merged_test = neg_pairs_test + pos_pairs_test
        return merged_train, merged_test

    def _merge_prep_features(self, feature_dict):
        neg_pairs = [feature_dict[0][ind] for ind in self.prep_neg_pairs]
        pos_pairs = [feature_dict[1][ind] for ind in self.prep_pos_pairs]
        merged_features = neg_pairs + pos_pairs
        return merged_features

    @staticmethod
    def _prepare_dataset(dataset_dict, file_type, merged_features, merged_labels):
        combined = list(zip(merged_features, merged_labels))
        np.random.shuffle(combined)
        dataset_dict[file_type]['X'] = np.array([elem[0] for elem in combined])
        dataset_dict[file_type]['Y'] = np.array([elem[1] for elem in combined])
        return dataset_dict

    def _train_and_evaluate(self, rel_dataset_dict, prep_mode=False):
        if self.evaluation_mode == 'blocking' and prep_mode is False:
            return
        params_dict = self._read_config_models(prep_mode)
        load_trained_models = config.Models.load_trained_models
        cv = config.Models.cv
        flexible_classifier = FlexibleClassifier(rel_dataset_dict, params_dict, self.seed,
                                                 self.logger, prep_mode, load_trained_models, cv)
        return flexible_classifier

    @staticmethod
    def _read_config_models(prep_mode):
        model_list = config.Models.model_list if not prep_mode else config.Models.prep_model_list
        params_dict = dict()
        for model in model_list:
            params_dict[model] = config.Models.params_dict[model]
        return params_dict
