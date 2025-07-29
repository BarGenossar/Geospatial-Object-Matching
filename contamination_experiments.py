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
from multiprocessing import Pool, cpu_count
import argparse
from pipelines import PipelineManager
import warnings
import copy



class ContaminationPipelineManager:
    def __init__(self, seed, logger, args):
        self.dataset_name = args.dataset_name
        self.seed = seed
        self.logger = logger
        self.dataset_size_version = args.dataset_size_version
        self.neg_samples_num = args.neg_samples_num
        self.vector_normalization = args.vector_normalization
        self.sdr_factor = args.sdr_factor
        self.contamination_levels = args.contamination_levels
        self.dataset_dict = self._read_dataset_dict()
        self.contaminated_result_dicts = self._run_contaminated_matching_pipeline_wrapper()

    def _read_dataset_dict(self):
        dataset_dict_path = self._get_dataset_dict_path()
        try:
            dataset_dict = joblib.load(dataset_dict_path)
            self.logger.info(f"dataset_dict was loaded successfully")
            return dataset_dict
        except Exception as e:
            self.logger.error(f"Error happened while loading dataset_dict: {e}")
            self.logger.error(f"First, run the regular matching experiments to generate the dataset_dict file")
            return None

    def _get_dataset_dict_path(self):
        dataset_dict_dir = config.FilePaths.dataset_dict_path
        file_name = get_file_name()
        if not os.path.exists(dataset_dict_dir):
            os.makedirs(dataset_dict_dir)
        dataset_dict_path = (f"{dataset_dict_dir}{file_name}_matching_{self.dataset_size_version}_"
                             f"neg_samples={self.neg_samples_num}_seed={self.seed}.joblib")
        return dataset_dict_path

    def _run_contaminated_matching_pipeline_wrapper(self):
        contaminated_result_dicts = defaultdict(dict)
        for contamination_level in self.contamination_levels:
            contamination_level = round(contamination_level, 2)
            self.logger.info(f"Running matching pipeline with contamination level: {contamination_level, 2}")
            contaminated_dataset_dict, contaminated_indices_dict = self._inject_contamination(contamination_level)
            flexible_classifier_obj = self._run_contaminated_matching_pipeline(contaminated_dataset_dict,
                                                                               contamination_level,
                                                                               contaminated_indices_dict)
            contaminated_result_dicts[contamination_level] = flexible_classifier_obj.result_dict
            self.logger.info("==========================================================")
        return contaminated_result_dicts

    def _inject_contamination(self, contamination_level):
        np.random.seed(self.seed)
        contaminated_dataset_dict = copy.deepcopy(self.dataset_dict)
        contaminated_indices_dict = dict()
        for train_or_test in contaminated_dataset_dict.keys():
            curr_X = contaminated_dataset_dict[train_or_test]['X']
            n_samples = curr_X.shape[0]
            n_contaminate = int(n_samples * contamination_level)
            contaminated_indices = np.random.choice(n_samples, n_contaminate, replace=False)
            X_contaminated = curr_X[contaminated_indices]
            mask = X_contaminated != 0
            recip = np.zeros_like(X_contaminated)
            recip[mask] = 1.0 / X_contaminated[mask]
            X_contaminated = np.minimum(recip, 1000)
            curr_X[contaminated_indices] = X_contaminated
            contaminated_dataset_dict[train_or_test]['X'] = curr_X
            contaminated_indices_dict[train_or_test] = contaminated_indices
        self.logger.info(f"Contamination level {contamination_level} injected successfully")
        return contaminated_dataset_dict, contaminated_indices_dict

    def _run_contaminated_matching_pipeline(self, contaminated_dataset_dict, contamination_level,
                                            contaminated_indices_dict):
        self.logger.info("Training for matching")
        params_dict = self._read_config_models_contamination()
        load_trained_models = False
        cv = config.Models.cv
        flexible_classifier_obj = FlexibleClassifier(contaminated_dataset_dict, None, params_dict, self.seed, self.logger,
                                                     self.dataset_name, 'matching', self.dataset_size_version,
                                                     self.neg_samples_num, load_trained_models, cv,
                                                     contamination_level, contaminated_indices_dict)
        return flexible_classifier_obj

    @staticmethod
    def _read_config_models_contamination():
        model_list = config.Models.contamination_model_list
        params_dict = dict()
        for model in model_list:
            params_dict[model] = config.Models.params_dict[model]
        return params_dict


def generate_final_contamination_results(result_dicts, args):
    results_path = config.FilePaths.results_path + f"matching csv files/contaminated"
    dataset_size_version = args.dataset_size_version
    neg_samples_num = args.neg_samples_num
    vector_normalization = args.vector_normalization
    blocking_method = args.blocking_method
    file_name = get_file_name(blocking_method)
    vector_normalization_str = "True" if vector_normalization else "False"
    file_path = (f"{results_path}FinalResults_{file_name}_matching_{dataset_size_version}_"
                 f"neg_samples={neg_samples_num}_vector_normalization={vector_normalization_str}")
    for contamination_level in args.contamination_levels:
        generate_contamination_level_df(result_dicts, contamination_level, file_path)
    return


def generate_contamination_level_df(result_dicts, contamination_level, file_path):
    file_path += f"_contamination={str(contamination_level)}.csv"
    final_res_dict = {}
    for model_name, model_dict in result_dicts[1][contamination_level].items():
        final_res_dict[model_name] = {}
        for metric in model_dict.keys():
            metric_res_list = []
            for seed in result_dicts.keys():
                metric_res_list.append(result_dicts[seed][contamination_level][model_name][metric])
            final_res_dict[model_name][metric] = round(np.mean(metric_res_list), 3)
    df = pd.DataFrame.from_dict(final_res_dict, orient='index')
    df.to_csv(file_path)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=config.Constants.dataset_name)
    parser.add_argument('--evaluation_mode', type=str, default=config.Constants.evaluation_mode)
    parser.add_argument('--run_preparatory_phase', type=bool, default=config.TrainingPhase.run_preparatory_phase)
    parser.add_argument('--blocking_method', type=str, default=config.Blocking.blocking_method)
    parser.add_argument('--seed_num', type=int, default=config.Constants.seeds_num)
    parser.add_argument('--dataset_size_version', type=str, default=config.Constants.dataset_size_version)
    parser.add_argument('--vector_normalization', default=True)
    parser.add_argument('--bkafi_criterion', type=str, default=config.Blocking.bkafi_criterion)
    parser.add_argument('--sdr_factor', default=False)
    parser.add_argument('--neg_samples_num', type=int, default=config.Constants.neg_samples_num)
    parser.add_argument('--contamination_levels', type=list, default=config.Constants.contamination_levels)

    args = parser.parse_args()
    logger = define_logger()
    print_config(logger, args)
    result_dicts = {}
    for seed in range(1, args.seed_num+1):
        logger.info(f"Seed: {seed}")
        logger.info(3*'--------------------------')
        pipeline_manager_obj = ContaminationPipelineManager(seed, logger, args)
        result_dicts[seed] = pipeline_manager_obj.contaminated_result_dicts
    generate_final_contamination_results(result_dicts, args)
    logger.info("Done!")
