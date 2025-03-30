from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, BaggingClassifier)
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
import logging
from sklearn.neural_network import MLPClassifier
import joblib
from collections import defaultdict
import config
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import get_feature_name_list, get_file_name


class FlexibleClassifier:
    def __init__(self, dataset_dict, property_dict, params_dict, seed, logger, evaluation_mode,
                 dataset_size_version, neg_samples_num, load_trained_models=False, cv=5):
        self.dataset_dict = dataset_dict
        self.property_dict = property_dict
        self.params_dict = params_dict
        self.seed = seed
        self.dataset_size_version = dataset_size_version
        self.neg_samples_num = neg_samples_num
        self.load_trained_models = load_trained_models
        self.cv = cv
        self.models_path = config.FilePaths.saved_models_path
        self.logger = logger
        self.evaluation_mode = evaluation_mode
        self.model_dict = self._get_model_dict()
        self.file_name = get_file_name()
        self.scorer = make_scorer(f1_score, average='macro')
        self.best_model_dict, self.result_dict = self._train_and_evaluate_all_models()
        self._print_results()

    def _get_model_dict(self):
        model_dict = {
            'RandomForestClassifier': RandomForestClassifier(random_state=self.seed),
            'SVC': SVC(random_state=self.seed),
            'LogisticRegression': LogisticRegression(random_state=self.seed),
            'MLPClassifier': MLPClassifier(random_state=self.seed),
            'AdaBoostClassifier': AdaBoostClassifier(random_state=self.seed),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=self.seed),
            'BaggingClassifier': BaggingClassifier(random_state=self.seed),
            'XGBClassifier': XGBClassifier(random_state=self.seed)
        }
        return model_dict

    def _save_model(self, model, model_name):
        general_file_name = (f"{self.evaluation_mode}_{self.file_name}_{self.dataset_size_version}_"
                             f"neg_samples_num={self.neg_samples_num}")
        if not os.path.exists(self.models_path[:-1]):
            os.makedirs(self.models_path[:-1])
        try:
            model_file_name = f'{self.models_path}{model_name}_{general_file_name}_seed{self.seed}.joblib'
            feature_name_list = self._get_final_feature_name_list()
            joblib.dump({'model': model, 'feature_name_list': feature_name_list}, model_file_name)
            logging.info(f"Model {model_name} was saved successfully ({self.evaluation_mode})")
            logging.info('')
        except Exception as e:
            logging.error(f"Error happened while saving model {model_name} ({self.evaluation_mode}): {e}")

    def _load_model(self, model_name):
        general_file_name = (f"{self.evaluation_mode}_{self.file_name}_{self.dataset_size_version}_"
                             f"neg_samples_num={self.neg_samples_num}")
        try:
            model = joblib.load(f'{self.models_path}{model_name}_{general_file_name}_seed{self.seed}.joblib')
            logging.info(f"Model {model_name} was loaded successfully")
            logging.info('')
            print(f"Model {model_name} was loaded successfully {self.evaluation_mode})")
            return model['model']
        except Exception as e:
            logging.error(f"Error happened while loading model {model_name}: {e}. Starting training...")
            print(f"Error happened while loading model {model_name}: {e}. Starting training...")
            return None

    @staticmethod
    def _get_final_feature_name_list():
        operator = config.Features.operator
        feature_name_list = get_feature_name_list(operator)
        # shape_features = config.Features.shape
        # neighborhood = self._get_neighborhood_feature_name_list()
        # roads = self._get_roads_feature_name_list()
        # feature_name_list = shape_features + neighborhood + roads
        return feature_name_list

    @staticmethod
    def _get_neighborhood_feature_name_list():
        if config.Features.knn_buildings == 0:
            return []
        with_knn = [f"{i}_nearest_building_dists" for i in range(1, config.Features.knn_buildings + 1)]
        other_features = [feature for feature in config.Features.neighborhood if feature != "nearest_building_dists"]
        return with_knn + other_features

    @staticmethod
    def _get_roads_feature_name_list():
        if config.Features.knn_roads == 0:
            return []
        with_knn = [f"{i}_nearest_road_dists" for i in range(1, config.Features.knn_roads + 1)]
        other_features = [feature for feature in config.Features.roads if feature != "nearest_road_dists"]
        return with_knn + other_features

    def _train_and_evaluate_all_models(self):
        result_dict = defaultdict(dict)
        best_model_dict = defaultdict(dict)
        for model_name, model_params in self.params_dict.items():
            try:
                best_model_dict[model_name], result_dict = self._train_and_evaluate_model(result_dict,
                                                                                          model_name,
                                                                                          model_params)
            except Exception as e:
                logging.error(f"Error for model {model_name}: {e}")
        return best_model_dict, result_dict

    def _get_best_model(self, model_name, params):
        if self.load_trained_models:
            best_model = self._load_model(model_name)
            if best_model is not None:
                return best_model
        model = self.model_dict[model_name]
        best_model = self._train_model(model_name, model, params)
        self._save_model(best_model, model_name)
        return best_model

    def _train_and_evaluate_model(self, result_dict, model_name, params):
        best_model = self._get_best_model(model_name, params)
        feature_name_list = self._get_final_feature_name_list()
        data_type = 'train' if self.evaluation_mode == "blocking" else 'test'
        x_test = self.dataset_dict[data_type]['X']
        y_test_preds = best_model.predict(x_test)
        result_dict = self._insert_results_to_dict(result_dict, model_name, y_test_preds)
        return {'model': best_model, 'feature_name_list': feature_name_list}, result_dict

    def _get_y_train(self, model_name):
        # y_train = self.dataset_dict['train']['Y'] if self.train_mode is not True else self.dataset_dict['prep']['Y']
        y_train = self.dataset_dict['train']['Y']
        if model_name == 'XGBClassifier':
            le = LabelEncoder()
            return le.fit_transform(y_train)
        else:
            return y_train

    def _train_model(self, model_name, model, params):
        if 'train' not in self.dataset_dict.keys():
            raise ValueError("You first need to run the code with "
                             "config.TrainingPhase.run_preparatory_phase = True")
        x_train = self.dataset_dict['train']['X']
        y_train = self._get_y_train(model_name)
        self.logger.info(f"Training model {model.__class__.__name__}...")
        grid_search = GridSearchCV(model, params, cv=self.cv, scoring=self.scorer)
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        return best_model

    def _insert_results_to_dict(self, result_dict, model_name, y_test_preds, y_prediction_file=None):
        # Y_test = self.dataset_dict['test']['Y'] if not self.train_mode else self.dataset_dict['prep']['Y']
        data_type = 'train' if self.evaluation_mode == "blocking" else 'test'
        y_test = self.dataset_dict[data_type]['Y']
        result_dict[model_name]['precision'] = precision_score(y_test, y_test_preds, average='binary')
        result_dict[model_name]['recall'] = recall_score(y_test, y_test_preds, average='binary')
        result_dict[model_name]['f1'] = f1_score(y_test, y_test_preds, average='binary')
        # result_dict[model_name]['confusion_matrix'] = confusion_matrix(Y_test, y_test_preds)
        return result_dict

    def _print_results(self):
        for model_name, model_results in self.result_dict.items():
            # self.logger.info(f"Results for model {model_name}{prep_mode_message}:")
            eval_mode_message = f" {self.evaluation_mode} mode (results over train set)" if (
                    self.evaluation_mode == "blocking") else ""
            self.logger.info(f"{eval_mode_message}")
            self.logger.info(f"Results for model {model_name}:")
            self.logger.info(f"Precision: {round(model_results['precision'], 3)}")
            self.logger.info(f"Recall: {round(model_results['recall'], 3)}")
            self.logger.info(f"F1 score: {round(model_results['f1'], 3)}")
            # self.logger.info(f"Confusion matrix: {model_results['confusion_matrix']}")
            self.logger.info(3*'--------------------------')
            self.logger.info('')
            # Display confusion matrix
            # disp = ConfusionMatrixDisplay(confusion_matrix=model_results['confusion_matrix'])
            # disp.plot()
            # plt.show()
        return

    def feature_importance_extraction(self):
        """
        Extracts the feature importance scores for the best model (used in the preparatory phase for blocking)
        """
        feature_importance_dict = dict()
        self.logger.info("Feature importance scores:\n")
        for model_name in self.best_model_dict.keys():
            best_model = self.best_model_dict[model_name]['model']
            feature_name_list = self.best_model_dict[model_name]['feature_name_list']
            sorted_importance_scores = sorted(zip(feature_name_list, best_model.feature_importances_),
                                              key=lambda x: x[1], reverse=True)
            self._print_feature_importance_scores(sorted_importance_scores)
            feature_importance_dict[model_name] = sorted_importance_scores
        self._save_feature_importance_scores(feature_importance_dict)
        self.logger.info(3 * '*******************************************')
        self.logger.info(3 * '*******************************************')
        return feature_importance_dict

    def _save_feature_importance_scores(self, sorted_importance_scores):
        general_file_name = ''.join((self.file_name, '_feature_importance_dict'))
        feature_importance_file_name = f'{self.models_path}_{general_file_name}_seed={self.seed}.joblib'
        joblib.dump(sorted_importance_scores, feature_importance_file_name)
        self.logger.info(f"Feature importance scores were saved successfully")
        self.logger.info('')
        return

    def _print_feature_importance_scores(self, sorted_importance_scores):
        for feature, score in sorted_importance_scores:
            self.logger.info(f"{feature}: {round(score, 3)}")
        self.logger.info(3 * '==============================')
        self.logger.info('')
        return

    def get_property_ratios(self):
        property_ratios = dict()
        for prop, curr_prop_dict in self.property_dict.items():
            ratio_hist = [curr_prop_dict['index'][ind] / curr_prop_dict['cands'][ind] for ind in
                          curr_prop_dict['index'].keys() if ind in curr_prop_dict['cands'].keys()]
            property_ratios[prop] = {'mean': round(np.mean(ratio_hist), 3),
                                     'std': round(np.std(ratio_hist), 3)}
        self._save_property_ratios(property_ratios)
        return property_ratios

    def _save_property_ratios(self, property_ratios):
        general_file_name = ''.join((self.file_name, '_property_ratios'))
        property_ratios_file_name = f'{self.models_path}{general_file_name}_seed={self.seed}.joblib'
        joblib.dump(property_ratios, property_ratios_file_name)
        self.logger.info(f"Matching pairs property ratios were saved successfully")
        self.logger.info('')
        return
