

class FilePaths:
    results_path = "results/"
    saved_models_path = "saved_model_files/"
    dataset_dict_path = "data/dataset_dicts/"
    property_dict_path = "data/property_dicts/"


class Constants:
    dataset_name = "gpkg"
    synthetic_folder_name = "example"  # Relevant only if dataset_name is "synthetic"
    evaluation_mode = "blocking"  # "blocking", "matching", "end2end"
    seeds_num = 1
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 1 - train_ratio - val_ratio
    max_ratio_val = 1000  # Avoid infinity values
    load_object_dict = False  # Load existing object dictionary
    save_object_dict = True  # Save the object dictionary
    load_prep_items = False  # Load existing preparatory items
    save_property_dict = True  # Save the properties dictionary
    load_property_dict = False  # Load the properties dictionary
    save_dataset_dict = True  # save the dataset dictionary
    load_dataset_dict = False  # load existing dataset dictionary
    file_name_suffix = "300125"  # If set to None, the current exact time will be used


class PreparatoryPhase:
    pos_pairs_num = 1000  # Number of positive samples
    neg_pairs_ratio = 1  # Number of negative samples per positive sample
    run_preparatory_phase = False  # If False, the preparatory phase will not be run
    load_pairs = False  # Load existing pairs


class Blocking:
    blocking_method = 'exhaustive'  # 'exhaustive', 'lsh', 'kdtree', 'bkafi'
    nn_param = 10  # number of nearest neighbors to retrieve as candidates
    cand_pairs = 2  # total number of candidate pairs to retrieve for each candidate object
    nbits = 10  # number of bits to use for LSH
    bkafi_dim = 3  # Number of important features to use for blocking (for the bkafi method)
    dist_threshold = None  # Define it as a hyperparameter or in a flexible manner


class Features:
    knn_buildings = 0  # Number of nearest buildings to consider
    knn_roads = 0  # Number of nearest roads to consider
    operator = 'division'  # 'division', 'concatenation'
    object_properties = ["bounding_box_width", "bounding_box_length", "area", "perimeter", "perimeter_ind",
                         "volume", "convex_hull_area", "convex_hull_volume", "ave_centroid_distance", "height_diff",
                         "num_floors", "axes_symmetry", "compactness_2d", "compactness_3d", "density",
                         "elongation", "shape_ind", "hemisphericality", "fractality", "cubeness", "circumference",
                         "aligned_bounding_box_width", "aligned_bounding_box_length", "aligned_bounding_box_height",
                         "num_vertices"]

    # object_properties = ["circumference", "density", "convex_hull_area"]

    neighborhood = []
    roads = []


class Models:
    load_trained_models = False
    cv = 3
    model_to_use = 'XGBClassifier'  # Used only for predict.py and feature_importances.py
    # model_list = ['RandomForestClassifier', 'SVC', 'LogisticRegression', 'AdaBoostClassifier',
    #               'GradientBoostingClassifier', 'BaggingClassifier', 'XGBClassifier']
    model_list = ['XGBClassifier']
    prep_model = ['XGBClassifier']  # Used only for preparatory phase. Use a list containing a single model
    params_dict = {
                    'RandomForestClassifier': {"n_estimators": [50, 100, 200],
                                               "max_depth": [5, 10],
                                               "min_samples_split": [2],
                                               "max_features": ["sqrt"]},

                    'SVC': {'C': [0.1, 0.5],
                           'kernel': ['rbf'],
                           'gamma': ['scale'],
                           'degree': [2]
                            },

                    'LogisticRegression': {'solver': ['lbfgs', 'saga'],
                                          'multi_class': ['auto'],
                                          'C': [0.01, 0.1, 1]
                                           },

                    'MLPClassifier': {'hidden_layer_sizes': [(128, 64), (128, 64, 32), (64, 32)],
                                      'activation': ['relu'],
                                      'solver': ['adam'],
                                      'batch_size': [16, 32],
                                      'max_iter': [500],
                                      },

                    'AdaBoostClassifier': {'n_estimators': [100, 200],
                                           'learning_rate': [0.1, 0.5, 1.0],
                                           'algorithm': ['SAMME']
                                           },

                    'GradientBoostingClassifier': {'loss': ['log_loss'],
                                                   'learning_rate': [0.01, 0.1],
                                                   'n_estimators': [100, 200],
                                                   'max_depth': [3],
                                                   'min_samples_split': [3],
                                                   'max_features': ['sqrt']
                                                   },

                    'BaggingClassifier': {'n_estimators': [10, 50],
                                          'max_samples': [0.5, 0.8, 1.0],
                                          'max_features': [0.5, 0.8],
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



