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
        self.dataset_size_version = args.dataset_size_version
        self.neg_samples_num = args.neg_samples_num
        self.vector_normalization = args.vector_normalization
        self.sdr_factor = args.sdr_factor
        self.matching_candidates_generation = args.matching_candidates_generation
        self.dataset_dict = self._create_dataset_dict()
        self.flexible_classifier_obj = self._train_and_evaluate()
        self.result_dict = self._get_result_dict()

    def _read_objects(self):
        dataset_config = json.load(open('dataset_configs.json'))[self.dataset_name]
        train_object_dict, test_object_dict = self._load_object_dict_wrapper()
        data_partition_dict = load_dataset_partition_dict(self.dataset_name, self.logger, self.seed)
        if test_object_dict is not None and train_object_dict is not None:
            return data_partition_dict, train_object_dict, test_object_dict
        self.logger.info("Generating test object dict and train object dict")
        object_dict = getattr(self, f'_read_objects_{self.dataset_name}')(dataset_config)
        train_object_dict, test_object_dict = self._partition_object_dict(object_dict, data_partition_dict)
        if config.Constants.save_object_dict:
            self._save_object_dicts(train_object_dict, test_object_dict, dataset_config)
        return data_partition_dict, train_object_dict, test_object_dict

    def _load_object_dict_wrapper(self):
        test_object_dict, train_object_dict = None, None
        train_full_path, test_full_path = self._get_object_dict_paths()
        if config.Constants.load_object_dict:
            train_object_dict = load_object_dict(self.logger, train_full_path, 'train_object_dict')
            test_object_dict = load_object_dict(self.logger, test_full_path, 'test_object_dict')
        return train_object_dict, test_object_dict

    def _save_object_dicts(self, train_object_dict, test_object_dict, dataset_config):
        self._print_object_dict_stats(train_object_dict, test_object_dict)
        self.logger.info(f"Saving test object dict")
        train_full_path, test_full_path = self._get_object_dict_paths()
        self.logger.info(f"Saving train object dict to {train_full_path}")
        joblib.dump(train_object_dict, train_full_path)
        self.logger.info(f"Saving test object dict to {test_full_path}")
        joblib.dump(test_object_dict, test_full_path)
        return

    @staticmethod
    def _print_object_dict_stats(train_object_dict, test_object_dict):
        print(f"Number of cands in train: {len(train_object_dict['cands'])}")
        print(f"Number of index in train: {len(train_object_dict['index'])}")
        print(f"Number of cands in test: {len(test_object_dict['cands'])}")
        print(f"Number of index in test: {len(test_object_dict['index'])}")

    def _get_object_dict_paths(self):
        object_dict_path = config.FilePaths.object_dict_path
        if not os.path.exists(object_dict_path):
            os.makedirs(object_dict_path)
        if self.evaluation_mode == 'blocking':
            train_full_path = f"{object_dict_path}train_blocking_{self.dataset_size_version}"
            test_full_path = f"{object_dict_path}test_blocking_{self.dataset_size_version}"
        else:
            train_full_path = (f"{object_dict_path}train_matching_{self.dataset_size_version}_"
                               f"neg_samples_num={self.neg_samples_num}")
            test_full_path = f"{object_dict_path}test_matching_{self.matching_candidates_generation}" \
                             f"_{self.dataset_size_version}_neg_samples_num={self.neg_samples_num}"
        return f"{train_full_path}_seed_{self.seed}.joblib", f"{test_full_path}_seed_{self.seed}.joblib"

    def _partition_object_dict(self, object_dict, data_partition_dict):
        if self.evaluation_mode == "blocking":
            return self._clean_object_dict_blocking(object_dict, data_partition_dict)
        else:
            return self._clean_object_dict_matching(object_dict, data_partition_dict)

    def _clean_object_dict_blocking(self, object_dict, data_partition_dict):
        dataset_version = self.dataset_size_version
        train_object_dict = {'cands': {}, 'index': {}}
        test_object_dict = {'cands': {}, 'index': {}}
        train_pairs = data_partition_dict['train'][dataset_version][2]
        test_data_partition = data_partition_dict['test']['blocking'][dataset_version]
        test_cands_ids = test_data_partition['cands']
        test_index_ids = test_data_partition['index']
        train_object_dict['cands'] = {pair[0]: object_dict['cands'][pair[0]] for pair in train_pairs}
        train_object_dict['index'] = {pair[1]: object_dict['index'][pair[1]] for pair in train_pairs}
        test_object_dict['cands'] = {object_id: object_dict['cands'][object_id] for object_id in test_cands_ids}
        test_object_dict['index'] = {object_id: object_dict['index'][object_id] for object_id in test_index_ids}
        return train_object_dict, test_object_dict

    def _clean_object_dict_matching(self, object_dict, data_partition_dict):
        train_object_dict = {'cands': {}, 'index': {}}
        test_object_dict = {'cands': {}, 'index': {}}
        dataset_version = self.dataset_size_version
        neg_num = self.neg_samples_num
        candidates_generation = self.matching_candidates_generation
        train_pairs = data_partition_dict['train'][dataset_version][neg_num]
        test_pairs = data_partition_dict['test']['matching'][candidates_generation][dataset_version][neg_num]
        train_object_dict['cands'] = {pair[0]: object_dict['cands'][pair[0]] for pair in train_pairs}
        train_object_dict['index'] = {pair[1]: object_dict['index'][pair[1]] for pair in train_pairs}
        test_object_dict['cands'] = {pair[0]: object_dict['cands'][pair[0]] for pair in test_pairs}
        test_object_dict['index'] = {pair[1]: object_dict['index'][pair[1]] for pair in test_pairs}
        return train_object_dict, test_object_dict

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
        for objects_type in objects_path_dict.keys():
            object_dict['mapping_dict'][objects_type] = {ind: obj_key for ind, obj_key in
                                                         enumerate(object_dict[objects_type].keys())}
            object_dict['inv_mapping_dict'][objects_type] = {obj_key: ind for ind, obj_key in
                                                             enumerate(object_dict[objects_type].keys())}
        return object_dict

    @staticmethod
    def read_objects_synthetic(self, dataset_config):
        pass

    # def _generate_training_pairs(self):
    #     np.random.seed(self.seed)
    #     index_ids = list(self.train_object_dict['index'].keys())
    #     pos_pairs = [(obj_id, obj_id) for obj_id in self.train_object_dict['cands'].keys()]
    #     neg_pairs = [(obj_id, np.random.choice(index_ids)) for obj_id in self.train_object_dict['cands'].keys()]
    #     neg_pairs = [(cand_id, index_id) for cand_id, index_id in neg_pairs if cand_id != index_id]
    #     return pos_pairs, neg_pairs

    # def _run_blocker(self):
    #     self.logger.info(f"Running blocking for training phase")
    #     dummy_feature_importance_scores = self._get_dummy_feature_importance_scores()
    #     dummy_property_ratios = self._get_dummy_property_ratios()
    #     blocker = Blocker(self.dataset_name, self.train_object_dict, self.train_property_dict,
    #                       dummy_feature_importance_scores, dummy_property_ratios, 'bkafi', 'train')
    #     self.logger.info(f"The blocking process for the training phase ended successfully")
    #     self.train_pos_pairs_dict, self.train_neg_pairs_dict = blocker.pos_pairs_dict, blocker.neg_pairs_dict
    #     save_blocking_output(self.train_pos_pairs_dict, self.train_neg_pairs_dict, self.seed, self.logger, 'train')
    #     return

    @staticmethod
    def _get_dummy_feature_importance_scores():
        feature_names = get_feature_name_list(config.Features.operator)
        dummy_model_name = config.Models.blocking_model
        return {dummy_model_name: [(feature, 1) for feature in feature_names]}

    def _get_dummy_property_ratios(self):
        property_ratios = {prop: {'mean': 1.0, 'std': 0.0}
                           for prop in self.train_property_dict.keys()}
        return property_ratios

    def _run_blocker(self, feature_importance_dict, train_property_ratios):
        blocking_method = self.blocking_method
        self.logger.info(f"Running blocking method {blocking_method}")
        blocker = Blocker(self.dataset_name, self.test_object_dict, self.test_property_dict, feature_importance_dict,
                          train_property_ratios,  self.blocking_method, self.sdr_factor, 'test')
        self.logger.info(f"The blocking process ended successfully")
        self._save_blocking_output(blocker.pos_pairs_dict, blocker.neg_pairs_dict, blocker.blocking_execution_time)
        self.blocking_result_dict = self._evaluate_blocking(blocker.pos_pairs_dict, blocker.blocking_execution_time)
        return

    def _save_blocking_output(self, pos_pairs, neg_pairs, blocking_execution_time):
        blocking_dict = {'pos_pairs': pos_pairs, 'neg_pairs': neg_pairs,
                         'blocking_execution_time': blocking_execution_time}
        blocking_output_path = self._get_blocking_output_path()
        try:
            joblib.dump(blocking_dict, blocking_output_path)
            self.logger.info(f"Blocking output wes saved successfully")
        except Exception as e:
            self.logger.error(f"Error happened while saving blocking results: {e}")
        return

    def _get_blocking_output_path(self):
        file_name = get_file_name()
        blocking_results_path = config.FilePaths.results_path
        vector_normalization = self.vector_normalization if self.vector_normalization is not None else 'None'
        sdr_factor = 'True' if self.sdr_factor else 'False'
        if not os.path.exists(blocking_results_path):
            os.makedirs(blocking_results_path)
        blocking_results_path = (f"{blocking_results_path}blocking_output_{file_name}_"
                                 f"{self.dataset_size_version}_neg_samples_num{self.neg_samples_num}"
                                 f"_vector_normalization_{vector_normalization}_sdr_factor_{sdr_factor}"
                                 f"_seed={self.seed}.joblib")
        return blocking_results_path


    # def _get_property_dict_path(self, train_or_test):
    #     file_name = get_file_name_property_dict()
    #     property_dict_path = config.FilePaths.property_dict_path
    #     vector_normalization = self.vector_normalization if self.vector_normalization is not None else 'None'
    #     if not os.path.exists(property_dict_path):
    #         os.makedirs(property_dict_path)
    #     property_dict_path = (f"{property_dict_path}{file_name}_{train_or_test}_{self.evaluation_mode}_"
    #                           f"{self.dataset_size_version}_neg_samples_num={self.neg_samples_num}_"
    #                           f"vector_normalization={vector_normalization}_seed={self.seed}.joblib")
    #     return property_dict_path

    def _save_blocking_evaluation(self, blocking_evaluation_dict, blocking_method_arg=None):
        file_name = get_file_name(blocking_method_arg)
        blocking_results_path = config.FilePaths.results_path
        if not os.path.exists(blocking_results_path):
            os.makedirs(blocking_results_path)
        try:
            blocking_results_path = (f"{blocking_results_path}blocking_evaluation_results_{file_name}_"
                                     f"{self.dataset_size_version}_neg_samples_num{self.neg_samples_num}_"
                                     f"seed={self.seed}.joblib")
            joblib.dump(blocking_evaluation_dict, blocking_results_path)
            self.logger.info(f"Blocking evaluation results were saved successfully")
        except Exception as e:
            self.logger.error(f"Error happened while saving blocking evaluation results: {e}")

    def _evaluate_blocking(self, pos_pairs_dict, blocking_execution_time):
        index_ids = set(self.test_object_dict['index'].keys())
        cand_ids = set(self.test_object_dict['cands'].keys())
        max_intersection = index_ids.intersection(cand_ids)
        if 'bkafi' in self.blocking_method:
            blocking_res_dict = self._evaluate_bkafi_blocking(max_intersection, pos_pairs_dict, blocking_execution_time)
        else:
            blocking_res_dict = self._evaluate_not_bkafi_blocking(max_intersection, pos_pairs_dict,
                                                                  blocking_execution_time)
        self._save_blocking_evaluation(blocking_res_dict)
        return blocking_res_dict

    def _evaluate_bkafi_blocking(self, max_intersection, pos_pairs_dict, blocking_execution_time):
        blocking_res_dict = defaultdict(dict)
        for bkafi_dim in pos_pairs_dict.keys():
            for cand_pairs_per_item in pos_pairs_dict[bkafi_dim].keys():
                pos_pairs = set(pos_pairs_dict[bkafi_dim][cand_pairs_per_item])
                blocking_recall = round(len(pos_pairs) / len(max_intersection), 3)
                blocking_res_dict[bkafi_dim][cand_pairs_per_item] = {'blocking_recall': blocking_recall,
                                                                     'blocking_execution_time':
                                                                         blocking_execution_time[bkafi_dim]}
                self.logger.info(f"Blocking recall for {self.blocking_method}_dim {bkafi_dim} and cand_pairs_per_item "
                                 f"{cand_pairs_per_item}: {blocking_recall}")
                self.logger.info(3*'- - - - - - - - - - - - -')
        return blocking_res_dict

    def _evaluate_not_bkafi_blocking(self, max_intersection, pos_pairs_dict, blocking_execution_time):
        blocking_res_dict = defaultdict(dict)
        for cand_pairs_per_item in pos_pairs_dict.keys():
            pos_pairs = set(pos_pairs_dict[cand_pairs_per_item])
            blocking_recall = round(len(pos_pairs) / len(max_intersection), 3)
            blocking_res_dict[cand_pairs_per_item] = {'blocking_recall': blocking_recall,
                                                      'blocking_execution_time': blocking_execution_time}
            self.logger.info(f"Blocking recall for {self.blocking_method}, cand_pairs_per_item "
                             f"{cand_pairs_per_item}: {blocking_recall}")
            self.logger.info(3*'--------------------------')
        return blocking_res_dict

    def _create_dataset_dict(self):
        dataset_dict = self._load_dataset_dict_wrapper()
        if dataset_dict is not None:
            return dataset_dict
        data_partition_dict, self.train_object_dict, self.test_object_dict = self._read_objects()
        self.train_pos_pairs, self.train_neg_pairs = self._extract_pairs(data_partition_dict, 'train')
        self.train_property_dict = self._generate_property_dict('train')
        if self.evaluation_mode == "matching":
            self.test_pos_pairs, self.test_neg_pairs = self._extract_pairs(data_partition_dict, 'test')
        self.test_property_dict = self._generate_property_dict('test')
        feature_dict = self._generate_feature_dict()
        dataset_dict = self._create_final_dict(feature_dict)
        return dataset_dict

    def _extract_pairs(self, data_partition_dict, train_or_test):
        if self.evaluation_mode == "blocking" or train_or_test == 'train':
            pair_list = data_partition_dict[train_or_test][self.dataset_size_version][self.neg_samples_num]
        else:
            pair_list = data_partition_dict[train_or_test]['matching'][self.matching_candidates_generation] \
                [self.dataset_size_version][self.neg_samples_num]
        pos_pairs = [pair for pair in pair_list if pair[0] == pair[1]]
        neg_pairs = [pair for pair in pair_list if pair[0] != pair[1]]
        return pos_pairs, neg_pairs

    def _load_dataset_dict_wrapper(self):
        dataset_dict = None
        if config.Constants.load_dataset_dict:
            dataset_dict = self._load_dataset_dict()
            if dataset_dict is not None:
                return dataset_dict
        return dataset_dict

    # def _get_pos_and_neg_pairs_for_training(self):
    #     bkafi_dim = min(self.train_pos_pairs_dict.keys())
    #     cand_pairs_per_item = min(self.test_pos_pairs_dict[bkafi_dim].keys())
    #     pos_pairs = self.train_pos_pairs_dict[bkafi_dim][cand_pairs_per_item]
    #     neg_pairs = self.train_neg_pairs_dict[bkafi_dim][cand_pairs_per_item]
    #     return pos_pairs, neg_pairs

    # def _get_pos_and_neg_pairs(self, train_or_test):
    #     pos_pairs_dict = self.test_pos_pairs_dict if train_or_test == 'test' else self.train_pos_pairs_dict
    #     neg_pairs_dict = self.test_neg_pairs_dict if train_or_test == 'test' else self.train_neg_pairs_dict
    #     bkafi_dim = min(pos_pairs_dict.keys())
    #     cand_pairs_per_item = min(pos_pairs_dict[bkafi_dim].keys())
    #     pos_pairs = pos_pairs_dict[bkafi_dim][cand_pairs_per_item]
    #     neg_pairs = neg_pairs_dict[bkafi_dim][cand_pairs_per_item]
    #     return pos_pairs, neg_pairs

    def _load_train_items(self):
        feature_importance_dict, matching_pairs_property_ratios = None, None
        try:
            feature_importance_dict = load_feature_importance_dict(self.seed, self.logger)
            matching_pairs_property_ratios = load_property_ratios(self.seed, self.logger)
        except:
            self.logger.info("Could not load training phase items. Running training phase pipeline")
        return feature_importance_dict, matching_pairs_property_ratios

    def _generate_property_dict(self, train_or_test):
        rel_object_dict = self.train_object_dict if train_or_test == 'train' else self.test_object_dict
        if config.Constants.load_property_dict:
            property_dict = self._load_property_dict(train_or_test)
            if property_dict is not None:
                return property_dict
        self.logger.info(f"Generating {train_or_test} property dictionary")
        property_dict = ObjectPropertiesProcessor(rel_object_dict, self.vector_normalization).prop_vals_dict
        if config.Constants.save_property_dict:
            self._save_property_dict(property_dict, train_or_test)
        return property_dict

    def _save_property_dict(self, property_dict, train_or_test):
        try:
            property_dict_path = self._get_property_dict_path(train_or_test)
            joblib.dump(property_dict, property_dict_path)
            self.logger.info(f"{train_or_test}_property_dict was saved successfully")
            self.logger.info('')
        except Exception as e:
            self.logger.error(f"Error happened while saving {train_or_test}_property_dict: {e}")
        return

    def _load_property_dict(self, train_or_test):
        property_dict_path = self._get_property_dict_path(train_or_test)
        try:
            property_dict = joblib.load(property_dict_path)
            self.logger.info(f"{train_or_test}_property_dict was loaded successfully")
            return property_dict
        except Exception as e:
            self.logger.error(f"Error happened while loading {train_or_test}_property_dict: {e}")
            return None

    def _get_property_dict_path(self, train_or_test):
        file_name = get_file_name_property_dict()
        property_dict_path = config.FilePaths.property_dict_path
        vector_normalization = self.vector_normalization if self.vector_normalization is not None else 'None'
        if not os.path.exists(property_dict_path):
            os.makedirs(property_dict_path)
        property_dict_path = (f"{property_dict_path}{file_name}_{train_or_test}_{self.evaluation_mode}_"
                              f"{self.dataset_size_version}_neg_samples_num={self.neg_samples_num}_"
                              f"vector_normalization={vector_normalization}_seed={self.seed}.joblib")
        return property_dict_path

    def _generate_feature_dict(self):
        feature_dict = {'train': {}, 'test': {}} if self.evaluation_mode == 'matching' else {'train': {}}
        for train_or_test in feature_dict.keys():
            pos_pairs, neg_pairs, property_dict = self._get_rel_pairs_and_property_dict(train_or_test)
            self.logger.info(f"Generating {train_or_test} feature vectors")
            for label, pairs_list in zip([0, 1], [neg_pairs, pos_pairs]):
                feature_dict[train_or_test][label] = PairProcessor(property_dict, pairs_list).feature_vec
        return feature_dict

    def _get_rel_pairs_and_property_dict(self, train_or_test):
        if train_or_test == 'train':
            pos_pairs, neg_pairs = self.train_pos_pairs, self.train_neg_pairs
            property_dict = self.train_property_dict
        else:
            pos_pairs, neg_pairs = self.test_pos_pairs, self.test_neg_pairs
            property_dict = self.test_property_dict
        return pos_pairs, neg_pairs, property_dict

    def _create_final_dict(self, feature_dict):
        np.random.seed(self.seed)
        dataset_dict = {'train': {}, 'test': {}} if self.evaluation_mode == 'matching' else {'train': {}}
        for train_or_test in dataset_dict.keys():
            merged_features, merged_labels = self._merge_features_and_labels(feature_dict, train_or_test)
            dataset_dict = self._prepare_dataset(dataset_dict, train_or_test, merged_features, merged_labels)
        if config.Constants.save_dataset_dict:
            self._save_dataset_dict(dataset_dict)
        return dataset_dict

    def _save_dataset_dict(self, dataset_dict):
        dataset_dict_path = self._get_dataset_dict_path()
        saving_message = f"dataset_dict was saved successfully"
        error_message = f"Error happened while saving dataset_dict: "
        try:
            joblib.dump(dataset_dict, dataset_dict_path)
            self.logger.info(saving_message)
            self.logger.info('')
        except Exception as e:
            self.logger.error(f"{error_message}{e}")
        return

    def _load_dataset_dict(self):
        dataset_dict_path = self._get_dataset_dict_path()
        try:
            dataset_dict = joblib.load(dataset_dict_path)
            self.logger.info(f"dataset_dict was loaded successfully")
            return dataset_dict
        except Exception as e:
            self.logger.error(f"Error happened while loading dataset_dict: {e}")
            return None

    def _get_dataset_dict_path(self):
        dataset_dict_dir = config.FilePaths.dataset_dict_path
        file_name = get_file_name()
        if not os.path.exists(dataset_dict_dir):
            os.makedirs(dataset_dict_dir)
        dataset_dict_path = (f"{dataset_dict_dir}{file_name}_{self.evaluation_mode}_{self.dataset_size_version}_"
                             f"neg_samples={self.neg_samples_num}_seed={self.seed}.joblib")
        return dataset_dict_path

    def _merge_features_and_labels(self, feature_dict, train_or_test):
        neg_feature_vecs, pos_feature_vecs = feature_dict[train_or_test][0], feature_dict[train_or_test][1]
        merged_features = neg_feature_vecs + pos_feature_vecs
        merged_labels = [0] * len(neg_feature_vecs) + [1] * len(pos_feature_vecs)
        return merged_features, merged_labels

    @staticmethod
    def _prepare_dataset(dataset_dict, file_type, merged_features, merged_labels):
        combined = list(zip(merged_features, merged_labels))
        np.random.shuffle(combined)
        dataset_dict[file_type]['X'] = np.array([elem[0] for elem in combined])
        dataset_dict[file_type]['Y'] = np.array([elem[1] for elem in combined])
        return dataset_dict

    def _train_and_evaluate(self, ):
        if self.evaluation_mode == 'blocking':
            feature_importance_dict, train_property_ratios = self._train_for_blocking()
            self._run_blocker(feature_importance_dict, train_property_ratios)
            flexible_classifier = None
        else:
            flexible_classifier = self._run_matching_pipeline()
        return flexible_classifier

    def _train_for_blocking(self):
        self.logger.info("Training for blocking")
        if config.Constants.load_train_items:
            feature_importance_dict, matching_pairs_property_ratios = self._load_train_items()
            if feature_importance_dict is not None and matching_pairs_property_ratios is not None:
                return feature_importance_dict, matching_pairs_property_ratios
        params_dict = self._read_config_models()
        load_trained_models = config.Models.load_trained_models
        cv = config.Models.cv
        flexible_classifier_obj = FlexibleClassifier(self.dataset_dict, self.train_property_dict, params_dict,
                                                     self.seed, self.logger, 'blocking',
                                                     self.dataset_size_version, self.neg_samples_num,
                                                     load_trained_models, cv)
        feature_importance_dict = flexible_classifier_obj.feature_importance_extraction()
        train_property_ratios = flexible_classifier_obj.get_property_ratios()
        return feature_importance_dict, train_property_ratios

    def _run_matching_pipeline(self):
        self.logger.info("Training for matching")
        params_dict = self._read_config_models()
        load_trained_models = config.Models.load_trained_models
        cv = config.Models.cv
        flexible_classifier_obj = FlexibleClassifier(self.dataset_dict, None, params_dict, self.seed,
                                                     self.logger, 'matching', self.dataset_size_version,
                                                     self.neg_samples_num, load_trained_models, cv)
        return flexible_classifier_obj

    # property_dict_path = (f"{property_dict_path}{file_name}_{train_or_test}_{self.evaluation_mode}_"
    #                       f"{self.dataset_size_version}_neg_samples_num={self.neg_samples_num}_"
    #                       f"seed={self.seed}.joblib")

    def _read_config_models(self):
        model_list = config.Models.model_list if self.evaluation_mode == 'matching' else [config.Models.blocking_model]
        params_dict = dict()
        for model in model_list:
            params_dict[model] = config.Models.params_dict[model]
        return params_dict

    def _get_result_dict(self):
        if self.evaluation_mode == 'blocking':
            return {'blocking': self.blocking_result_dict}
        elif self.evaluation_mode == 'matching':
            return {'matching': self.flexible_classifier_obj.result_dict}
        else:
            raise ValueError(f"Evaluation mode {self.evaluation_mode} is not supported")
