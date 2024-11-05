

class FilePaths:
    results_path = "results/"
    saved_models_path = "saved_model_files/"
    dataset_dict_path = "data/dataset_dicts/"


class Constants:
    model = "classic"
    dataset_name = "gpkg"
    seeds_num = 3
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 1 - train_ratio - val_ratio
    max_ratio_val = 1000  # Avoid infinity values
    load_dataset_dict = False  # load existing dataset dictionary
    save_dataset_dict = True  # save the dataset dictionary


class Blocking:
    nn_param = 10  # number of nearest neighbors to retrieve as candidates
    cand_pairs = 2  # total number of candidate pairs to retrieve for each candidate object
    dist_threshold = None  # Define it as a hyperparameter or in a flexible manner


class Features:
    knn_buildings = 0  # Number of nearest buildings to consider
    knn_roads = 0  # Number of nearest roads to consider
    object_properties = ["bounding_box_width", "bounding_box_length", "area", "perimeter", "perimeter_index",
                         "convex_hull_area", "ave_centroid_distance", "height_diff", "num_floors", "axes_symmetry",
                         "volume", "density", "elongation", "shape_index", "hemisphericality", "fractality", "cubeness",
                         "circumference", "aligned_bounding_box_width", "aligned_bounding_box_length",
                         "aligned_bounding_box_height", "num_vertices",
                         ]

    # object_properties = ["circumference", "density", "convex_hull_area"]

    neighborhood = []
    roads = []


class Models:
    load_trained_models = False
    cv = 3
    model_to_use = 'RandomForestClassifier'  # Used only for predict.py and feature_importances.py
    model_list = ['RandomForestClassifier']
    params_dict = {
                    'RandomForestClassifier': {"n_estimators": [100],
                                               "max_depth": [10],
                                               "min_samples_split": [2],
                                               "max_features": ["sqrt"]},

                    'SVC': {'C': [0.1, 1, 10, 100],
                           'kernel': ['linear', 'rbf'],
                           'gamma': ['scale', 'auto', 0.1, 1, 10],
                           'degree': [2, 3, 4]
                            },

                    'LogisticRegression': {'solver': ['lbfgs', 'saga'],
                                          'multi_class': ['auto', 'ovr', 'multinomial'],
                                          'C': [0.001, 0.01, 0.1, 1, 10, 100]
                                           },

                    'MLPClassifier': {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                                      'activation': ['relu'],
                                      'solver': ['adam'],
                                      'batch_size': [20, 40],
                                      'max_iter': [800, 1000],
                                      },

                    'AdaBoostClassifier': {'n_estimators': [50, 100, 150, 200],
                                           'learning_rate': [0.1, 0.5, 1.0],
                                           'algorithm': ['SAMME', 'SAMME.R']
                                           },

                    'GradientBoostingClassifier': {'loss': ['log_loss', 'exponential'],
                                                   'learning_rate': [0.01, 0.1, 0.5],
                                                   'n_estimators': [50, 100, 150, 200],
                                                   'max_depth': [3, 5],
                                                   'min_samples_split': [2, 4, 5],
                                                   'max_features': ['sqrt', 'log2']
                                                   },

                    'BaggingClassifier': {'n_estimators': [10, 20, 50],
                                          'max_samples': [0.5, 0.8, 1.0],
                                          'max_features': [0.5, 0.8, 1.0],
                                          'bootstrap': [True, False]
                                          },

                    'XGBClassifier': {'max_depth': [3, 4, 5],
                                      'objective': ['binary:logistic'],
                                      'learning_rate': [0.01, 0.01, 0.1, 0.5],
                                      'n_estimators': [50, 100, 150, 200],
                                      'gamma': [0, 0.1, 1],
                                      # 'min_child_weight': [1, 5, 10],
                                      # 'reg_alpha': [0, 0.1, 1],
                                      # 'base_score': [0.5, 0.8, 1.0]
                                      }
    }



