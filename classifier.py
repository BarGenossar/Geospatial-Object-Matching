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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import get_feature_name_list, get_file_name


class FlexibleClassifier:
    def __init__(self, dataset_dict, params_dict, seed, logger, load_trained_models=False, cv=5):
        self.dataset_dict = dataset_dict
        self.params_dict = params_dict
        self.seed = seed
        self.load_trained_models = load_trained_models
        self.cv = cv
        self.models_path = config.FilePaths.models_path
        self.logger = logger
        self.model_dict = self._get_model_dict()
        self.file_name = get_file_name()
        self.scorer = make_scorer(f1_score, average='macro')
        self.result_dict = self._train_and_evaluate_all_models()
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

    def save_model(self, model, model_name):
        if not os.path.exists(self.models_path[:-1]):
            os.makedirs(self.models_path[:-1])
        try:
            model_file_name = f'{self.models_path}{model_name}_{self.file_name}_seed{self.seed}.joblib'
            feature_name_list = self._get_final_feature_name_list()
            joblib.dump({'model': model, 'feature_name_list': feature_name_list}, model_file_name)
            logging.info(f"Model {model_name} was saved successfully")
            logging.info('')
        except Exception as e:
            logging.error(f"Error happened while saving model {model_name}: {e}")

    def load_model(self, model_name):
        try:
            model = joblib.load(f'{self.models_path}{model_name}_{self.file_name}_seed{self.seed}.joblib')
            logging.info(f"Model {model_name} was loaded successfully")
            logging.info('')
            return model['model']
        except Exception as e:
            logging.error(f"Error happened while loading model {model_name}: {e}")
            print(f"Error happened while loading model {model_name}: {e}")
            return None

    def _get_final_feature_name_list(self):
        feature_name_list = get_feature_name_list()
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
        for model_name, model_params in self.params_dict.items():
            try:
                result_dict = self._train_and_evaluate_model(result_dict, model_name, model_params)
            except Exception as e:
                logging.error(f"Error for model {model_name}: {e}")
        return result_dict

    def _get_best_model(self, model_name, params):
        if self.load_trained_models:
            best_model = self.load_model(model_name)
        else:
            model = self.model_dict[model_name]
            best_model = self._train_model(model_name, model, params)
            self.save_model(best_model, model_name)
        return best_model

    def _train_and_evaluate_model(self, result_dict, model_name, params):
        best_model = self._get_best_model(model_name, params)
        y_test_preds = best_model.predict(self.dataset_dict['test']['X'])
        # y_prediction_file = best_model.predict(self.dataset_dict['prediction']['X'])
        result_dict = self._insert_results2dict(result_dict, model_name, y_test_preds)
        return result_dict

    def _get_y_train(self, model_name):
        if model_name == 'XGBClassifier':
            le = LabelEncoder()
            return le.fit_transform(self.dataset_dict['train']['Y'])
        else:
            return self.dataset_dict['train']['Y']

    def _train_model(self, model_name, model, params):
        self.logger.info(f"Training model {model.__class__.__name__}...")
        grid_search = GridSearchCV(model, params, cv=self.cv, scoring=self.scorer)
        y_train = self._get_y_train(model_name)
        grid_search.fit(self.dataset_dict['train']['X'], y_train)
        best_model = grid_search.best_estimator_
        return best_model

    def _insert_results2dict(self, result_dict, model_name, y_test_preds, y_prediction_file=None):
        # result_dict[model_name]['predictions'] = y_prediction_file
        Y_test = self.dataset_dict['test']['Y']
        result_dict[model_name]['precision'] = precision_score(Y_test, y_test_preds, average='macro')
        result_dict[model_name]['recall'] = recall_score(Y_test, y_test_preds, average='macro')
        result_dict[model_name]['f1'] = f1_score(Y_test, y_test_preds, average='macro')
        result_dict[model_name]['confusion_matrix'] = confusion_matrix(Y_test, y_test_preds)
        # Print the ids of all misclassified samples, divided into false positives and false negatives. Then,
        # put these ids in a txt file and save it in the results folder
        false_positives = [ind for ind in range(len(Y_test)) if Y_test[ind] == 0 and y_test_preds[ind] == 1]
        false_negatives = [ind for ind in range(len(Y_test)) if Y_test[ind] == 1 and y_test_preds[ind] == 0]
        # put the ids of the misclassified samples in a txt file, divided into false positives and false negatives
        # with open(f'{config.FilePaths.results_path}fp_{model_name}_{self.file_name}_seed{self.seed}.txt', 'w') as f:
        #     for item in false_positives:
        #         f.write(f"{item}\n")
        # with open(f'{config.FilePaths.results_path}fn_{model_name}_{self.file_name}_seed{self.seed}.txt', 'w') as f:
        #     for item in false_negatives:
        #         f.write(f"{item}\n")
        return result_dict

    def _print_results(self):
        for model_name, model_results in self.result_dict.items():
            self.logger.info(f"Model name: {model_name}")
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