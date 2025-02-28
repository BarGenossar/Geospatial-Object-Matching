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
    def __init__(self, seed, logger, args, min_surfaces_num=10):
        self.dataset_name = args.dataset_name
        self.seed = seed
        self.logger = logger
        self.with_prep_training = args.run_preparatory_phase
        self.min_surfaces_num = min_surfaces_num
        self.evaluation_mode = args.evaluation_mode
        self.blocking_method = args.blocking_method
        self.test_object_dict = None
        self.pos_pairs, self.neg_pairs = None, None
        # self.neg_indices_train, self.neg_indices_test = None, None
        # self.pos_indices_train, self.pos_indices_test = None, None
        self.train_feature_importance_scores, self.train_property_ratios = None, None
        self.dataset_dict = self._create_dataset_dict()
        self.flexible_classifier_obj = self._train_and_evaluate(self.dataset_dict, train_or_test='test')
        self.result_dict = self._get_result_dict()

    def _read_objects(self):
        dataset_config = json.load(open('dataset_configs.json'))[self.dataset_name]
        test_object_dict, train_object_dict = self._load_object_dict_wrapper(dataset_config)
        if test_object_dict is not None and train_object_dict is not None:
            return test_object_dict, train_object_dict
        self.logger.info("Generating test object dict and train object dict")
        test_object_dict = getattr(self, f'_read_objects_{self.dataset_name}')(dataset_config)
        test_object_dict, train_object_dict = self._get_training_object_ids(test_object_dict)
        if config.Constants.save_object_dict:
            self._save_object_dicts(test_object_dict, train_object_dict, dataset_config)
        return test_object_dict, train_object_dict

    def _load_object_dict_wrapper(self, dataset_config):
        test_object_dict, train_object_dict = None, None
        object_dict_path = dataset_config['object_dict_path']
        train_object_dict_path = object_dict_path.replace('test', 'train')
        if config.Constants.load_object_dict:
            test_object_dict = load_object_dict(self.logger, object_dict_path, 'test_object_dict')
            train_object_dict = load_object_dict(self.logger, train_object_dict_path, 'train_object_dict')
        return test_object_dict, train_object_dict

    def _save_object_dicts(self, test_object_dict, train_object_dict, dataset_config):
        print(f"Number of cands in train: {len(train_object_dict['cands'])}")
        print(f"Number of index in train: {len(train_object_dict['index'])}")
        print(f"Number of cands in test: {len(test_object_dict['cands'])}")
        print(f"Number of index in test: {len(test_object_dict['index'])}")
        self.logger.info(f"Saving test object dict")
        joblib.dump(test_object_dict, dataset_config['object_dict_path'])
        self.logger.info(f"Saving train object dict")
        train_object_dict_path = dataset_config['object_dict_path'].replace('test', 'train')
        joblib.dump(train_object_dict, train_object_dict_path)
        return

    def _get_training_object_ids(self, test_object_dict):
        """
        Get the object ids for the training phase from cands by randomly selecting the ids from the
        object_dict['cands'].keys(). Then take them from the object_dict['index'].

        Parameters:
            object_dict (dict): The object dictionary containing the cands and index objects.

        Returns:

        """
        np.random.seed(self.seed)
        cands_ids = set(test_object_dict['cands'].keys())
        index_ids = set(test_object_dict['index'].keys())
        pos_samples_num = int(config.TrainingPhase.training_ratio * len(cands_ids))
        intersection_set = cands_ids.intersection(index_ids)
        train_ids = set(np.random.choice(list(cands_ids), pos_samples_num, replace=False))
        train_object_dict = test_object_dict.copy()
        train_object_dict['cands'] = {object_id: test_object_dict['cands'][object_id] for object_id in train_ids}
        test_object_dict = self._remove_train_objects_from_object_dict(test_object_dict, train_ids)
        return test_object_dict, train_object_dict

    @staticmethod
    def _remove_train_objects_from_object_dict(object_dict, train_ids):
        for objects_type in object_dict.keys():
            object_dict[objects_type] = {
                object_id: object_data
                for object_id, object_data in object_dict[objects_type].items()
                if object_id not in train_ids
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
                        polygon_mesh_data = self._get_polygon_mesh(data, obj_key, vertices)
                        if polygon_mesh_data is not None:
                            object_dict[objects_type][new_obj_key] = polygon_mesh_data
                    except:
                        continue
        intersection_keys = set(object_dict['cands'].keys()).intersection(set(object_dict['index'].keys()))
        object_dict['cands'] = {obj_key: object_dict['cands'][obj_key] for obj_key in intersection_keys}
        object_dict['index'] = {obj_key: object_dict['index'][obj_key] for obj_key in intersection_keys}
        # print the len of the objects
        print(f"Number of cands: {len(object_dict['cands'])}")
        print(f"Number of index: {len(object_dict['index'])}")
        for objects_type in objects_path_dict.keys():
            object_dict['mapping_dict'][objects_type] = {ind: obj_key for ind, obj_key in
                                                         enumerate(object_dict[objects_type].keys())}
            object_dict['inv_mapping_dict'][objects_type] = {obj_key: ind for ind, obj_key in
                                                             enumerate(object_dict[objects_type].keys())}
        return object_dict

    @staticmethod
    def read_objects_synthetic(self, dataset_config):
        pass

    # def _split_data(self):
    #     self.neg_indices_train, self.neg_indices_test = self._split_indices(0)
    #     self.pos_indices_train, self.pos_indices_test = self._split_indices(1)
    #     return

    # def _merge_labels(self):
    #     neg_label_train = [0] * len(self.neg_indices_train)
    #     neg_label_test = [0] * len(self.neg_indices_test)
    #     pos_label_train = [1] * len(self.pos_indices_train)
    #     pos_label_test = [1] * len(self.pos_indices_test)
    #     merged_train = neg_label_train + pos_label_train
    #     merged_test = neg_label_test + pos_label_test
    #     return merged_train, merged_test

    def _generate_training_pairs(self):
        np.random.seed(self.seed)
        index_ids = list(self.train_object_dict['index'].keys())
        pos_pairs = [(obj_id, obj_id) for obj_id in self.train_object_dict['cands'].keys()]
        neg_pairs = [(obj_id, np.random.choice(index_ids)) for obj_id in self.train_object_dict['cands'].keys()]
        neg_pairs = [(cand_id, index_id) for cand_id, index_id in neg_pairs if cand_id != index_id]
        return pos_pairs, neg_pairs

    def _run_blocker_training(self):
        self.logger.info(f"Running blocking for training phase")
        dummy_feature_importance_scores = self._get_dummy_feature_importance_scores()
        dummy_property_ratios = self._get_dummy_property_ratios()
        blocker = Blocker(self.train_object_dict, self.train_property_dict, dummy_feature_importance_scores,
                          dummy_property_ratios, 'bkafi', 'train')
        self.logger.info(f"The blocking process for the training phase ended successfully")
        self.train_pos_pairs_dict, self.train_neg_pairs_dict = blocker.pos_pairs_dict, blocker.neg_pairs_dict
        save_blocking_output(self.train_pos_pairs_dict, self.train_neg_pairs_dict, self.seed, self.logger, 'train')
        return

    @staticmethod
    def _get_dummy_feature_importance_scores():
        feature_names = get_feature_name_list(config.Features.operator)
        dummy_model_name = config.Models.blocking_model
        return {dummy_model_name: [(feature, 1) for feature in feature_names]}

    def _get_dummy_property_ratios(self):
        property_ratios = {prop: {'mean': 1.0, 'std': 0.0}
                           for prop in self.train_property_dict.keys()}
        return property_ratios

    def _run_blocker(self):
        blocking_method = config.Blocking.blocking_method
        self.logger.info(f"Running blocking method {blocking_method}")
        blocker = Blocker(self.test_object_dict, self.test_property_dict, self.train_feature_importance_scores,
                          self.train_property_ratios, self.blocking_method, 'test')
        self.logger.info(f"The blocking process ended successfully")
        self.test_pos_pairs_dict, self.test_neg_pairs_dict = blocker.pos_pairs_dict, blocker.neg_pairs_dict
        save_blocking_output(self.test_pos_pairs_dict, self.test_neg_pairs_dict, self.seed, self.logger, 'test')
        if self.evaluation_mode == 'blocking':
            self.result_dict = self._evaluate_blocking()
        return

    def _evaluate_blocking(self):
        index_ids = set(self.test_object_dict['index'].keys())
        cand_ids = set(self.test_object_dict['cands'].keys())
        max_intersection = index_ids.intersection(cand_ids)
        blocking_res_dict = defaultdict(dict)
        for bkafi_dim in self.test_pos_pairs_dict.keys():
            for cand_pairs_per_item in self.test_pos_pairs_dict[bkafi_dim].keys():
                pos_pairs = set(self.test_pos_pairs_dict[bkafi_dim][cand_pairs_per_item])
                neg_pairs = set(self.test_neg_pairs_dict[bkafi_dim][cand_pairs_per_item])
                blocking_recall = round(len(pos_pairs) / len(max_intersection), 3)
                total_pairs = len(pos_pairs) + len(neg_pairs)
                blocking_res_dict[bkafi_dim][cand_pairs_per_item] = {'blocking_recall': blocking_recall,
                                                                     'k': cand_pairs_per_item}
                self.logger.info(f"Blocking recall for bkafi_dim {bkafi_dim} and cand_pairs_per_item "
                                 f"{cand_pairs_per_item}: {blocking_recall}")
                self.logger.info(3*'- - - - - - - - - - - - -')
        save_blocking_evaluation(blocking_res_dict, self.seed, self.logger)
        return blocking_res_dict

    # def _split_indices(self, label):
    #     test_size = config.Constants.test_ratio
    #     pairs = self.neg_pairs if label == 0 else self.pos_pairs
    #     indices = list(range(len(pairs)))
    #     indices_train, indices_test = train_test_split(indices, test_size=test_size, random_state=label+self.seed)
    #     return set(indices_train), set(indices_test)

    def _create_dataset_dict(self):
        dataset_dict = self._load_dataset_dict_wrapper()
        if dataset_dict is not None:
            return dataset_dict
        self.test_object_dict, self.train_object_dict = self._read_objects()
        self.train_property_dict = self._generate_property_dict('train')
        self._run_blocker_training()
        self.train_pos_pairs, self.train_neg_pairs = self._get_pos_and_neg_pairs('train')
        self.train_feature_importance_scores, self.train_property_ratios, dataset_dict = self._run_train_pipeline()
        self.test_property_dict = self._generate_property_dict('test')
        self._run_blocker()
        if self.evaluation_mode == "blocking":
            return
        self.pos_pairs, self.neg_pairs = self._get_pos_and_neg_pairs('test')
        # self._split_data()
        feature_dict = self._generate_feature_dict()
        dataset_dict = self._create_final_dict(dataset_dict, feature_dict)
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

    def _get_pos_and_neg_pairs_for_training(self):
        bkafi_dim = min(self.train_pos_pairs_dict.keys())
        cand_pairs_per_item = min(self.test_pos_pairs_dict[bkafi_dim].keys())
        pos_pairs = self.train_pos_pairs_dict[bkafi_dim][cand_pairs_per_item]
        neg_pairs = self.train_neg_pairs_dict[bkafi_dim][cand_pairs_per_item]
        return pos_pairs, neg_pairs

    def _get_pos_and_neg_pairs(self, train_or_test):
        pos_pairs_dict = self.test_pos_pairs_dict if train_or_test == 'test' else self.train_pos_pairs_dict
        neg_pairs_dict = self.test_neg_pairs_dict if train_or_test == 'test' else self.train_neg_pairs_dict
        bkafi_dim = min(pos_pairs_dict.keys())
        cand_pairs_per_item = min(pos_pairs_dict[bkafi_dim].keys())
        pos_pairs = pos_pairs_dict[bkafi_dim][cand_pairs_per_item]
        neg_pairs = neg_pairs_dict[bkafi_dim][cand_pairs_per_item]
        return pos_pairs, neg_pairs

    def _run_train_pipeline(self):
        feature_importance_dict = None
        train_property_ratios = None
        if config.Constants.load_train_items:
            feature_importance_dict, matching_pairs_property_ratios = self._load_train_items()
            dataset_dict = load_dataset_dict(self.logger, self.seed)
            if (feature_importance_dict is not None and matching_pairs_property_ratios is not None
                    and dataset_dict is not None):
                return feature_importance_dict, matching_pairs_property_ratios, dataset_dict
        train_feature_dict = self._generate_train_feature_dict()
        dataset_dict = self._create_train_final_dict(train_feature_dict)
        if self.with_prep_training:
            flexible_classifier_obj = self._train_and_evaluate(dataset_dict, 'train')
            feature_importance_dict = flexible_classifier_obj.feature_importance_extraction()
            train_property_ratios = flexible_classifier_obj.get_property_ratios('train')
        return feature_importance_dict, train_property_ratios, dataset_dict

    def _load_train_items(self):
        feature_importance_dict, matching_pairs_property_ratios = None, None
        try:
            feature_importance_dict = load_feature_importance_dict(self.seed, self.logger)
            matching_pairs_property_ratios = load_property_ratios(self.seed, self.logger)
        except:
            self.logger.info("Could not load training phase items. Running training phase pipeline")
        return feature_importance_dict, matching_pairs_property_ratios

    # def _generate_test_property_dict(self):
    #     if config.Constants.load_property_dict:
    #         property_dict = load_property_dict(self.logger, self.seed, train_mode_name=False)
    #         if property_dict is not None:
    #             return property_dict
    #     self.logger.info("Generating property dictionary")
    #     obj_prop_vals = ObjectPropertiesProcessor(self.test_object_dict).prop_vals_dict
    #     if config.Constants.save_property_dict:
    #         save_property_dict(obj_prop_vals, self.seed, self.logger, train_mode_name=False)
    #     return obj_prop_vals

    def _generate_property_dict(self, train_or_test):
        rel_object_dict = self.train_object_dict if train_or_test == 'train' else self.test_object_dict
        if config.Constants.load_property_dict:
            property_dict = load_property_dict(self.logger, self.seed, train_or_test)
            if property_dict is not None:
                return property_dict
        self.logger.info(f"Generating {train_or_test} property dictionary")
        property_dict = ObjectPropertiesProcessor(rel_object_dict).prop_vals_dict
        if config.Constants.save_property_dict:
            save_property_dict(property_dict, self.seed, self.logger, train_or_test)
        return property_dict

    def _generate_feature_dict(self):
        feature_dict = defaultdict(dict)
        self.logger.info("Generating feature vectors")
        for label, pairs_list in zip([0, 1], [self.neg_pairs, self.pos_pairs]):
            feature_dict[label] = PairProcessor(self.test_property_dict, pairs_list).feature_vec
        return feature_dict

    def _generate_train_feature_dict(self):
        train_feature_dict = defaultdict(dict)
        self.logger.info("Generating training phase feature vectors")
        for label, pairs_list in zip([0, 1], [self.train_neg_pairs, self.train_pos_pairs]):
            train_feature_dict[label] = PairProcessor(self.train_property_dict, pairs_list).feature_vec
        return train_feature_dict

    def _create_final_dict(self, dataset_dict, feature_dict):
        np.random.seed(self.seed)
        # merged_features_train, merged_features_test = self._merge_features(feature_dict)
        # merged_labels_train, merged_labels_test = self._merge_labels()
        merged_features_test = feature_dict[0] + feature_dict[1]
        merged_labels_test = [0] * len(self.neg_pairs) + [1] * len(self.pos_pairs)
        # dataset_dict = self._prepare_dataset(dataset_dict, 'train', merged_features_train, merged_labels_train)
        dataset_dict = self._prepare_dataset(dataset_dict, 'test', merged_features_test, merged_labels_test)
        return dataset_dict

    def _create_train_final_dict(self, feature_dict):
        dataset_dict = defaultdict(dict)
        np.random.seed(self.seed)
        merged_features = feature_dict[0] + feature_dict[1]
        labels = [0] * len(self.train_neg_pairs) + [1] * len(self.train_pos_pairs)
        dataset_dict = self._prepare_dataset(dataset_dict, 'train', merged_features, labels)
        if config.Constants.save_dataset_dict:
            save_dataset_dict(dataset_dict, self.seed, self.logger)
        return dataset_dict

    # def _merge_features(self, feature_dict):
    #     neg_pairs_train = [feature_dict[0][ind] for ind in self.neg_indices_train]
    #     neg_pairs_test = [feature_dict[0][ind] for ind in self.neg_indices_test]
    #     pos_pairs_train = [feature_dict[1][ind] for ind in self.pos_indices_train]
    #     pos_pairs_test = [feature_dict[1][ind] for ind in self.pos_indices_test]
    #     merged_train = neg_pairs_train + pos_pairs_train
    #     merged_test = neg_pairs_test + pos_pairs_test
    #     return merged_train, merged_test

    def _merge_train_features(self, feature_dict):
        neg_pairs = [feature_dict[0][ind] for ind in self.train_neg_pairs]
        pos_pairs = [feature_dict[1][ind] for ind in self.train_pos_pairs]
        merged_features = neg_pairs + pos_pairs
        return merged_features

    @staticmethod
    def _prepare_dataset(dataset_dict, file_type, merged_features, merged_labels):
        combined = list(zip(merged_features, merged_labels))
        np.random.shuffle(combined)
        dataset_dict[file_type]['X'] = np.array([elem[0] for elem in combined])
        dataset_dict[file_type]['Y'] = np.array([elem[1] for elem in combined])
        return dataset_dict

    def _train_and_evaluate(self, rel_dataset_dict, train_or_test):
        # if self.evaluation_mode == 'blocking' and train_or_test == 'train':
        #     return None
        params_dict = self._read_config_models(train_or_test)
        load_trained_models = config.Models.load_trained_models
        cv = config.Models.cv
        flexible_classifier = FlexibleClassifier(rel_dataset_dict, params_dict, self.seed,
                                                 self.logger, train_or_test, load_trained_models, cv)
        return flexible_classifier

    @staticmethod
    def _read_config_models(train_or_test):
        model_list = config.Models.model_list if train_or_test == 'test' else [config.Models.blocking_model]
        params_dict = dict()
        for model in model_list:
            params_dict[model] = config.Models.params_dict[model]
        return params_dict

    def _get_result_dict(self):
        if self.evaluation_mode == 'blocking':
            return self.result_dict
        elif self.evaluation_mode == 'matching':
            return self.flexible_classifier_obj.result_dict
        elif self.evaluation_mode == 'end2end':
            return {'blocking': self.result_dict, 'matching': self.flexible_classifier_obj.result_dict}
        else:
            raise ValueError(f"Evaluation mode {self.evaluation_mode} is not supported")
