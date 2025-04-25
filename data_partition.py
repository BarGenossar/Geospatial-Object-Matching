import json
import os
import config
from utils import *
import pickle as pkl
import argparse
from time import time


class DataPartitionGenerator:
    def __init__(self, args, min_surfaces_num=10):
        self.dataset_name = args.dataset_name
        self.min_surfaces_num = min_surfaces_num
        self.train_neg_samples_list = args.train_neg_samples_list
        self.train_size_ratio_list = args.train_size_ratio_list
        self.test_size_ratio_list = args.test_size_ratio_list
        self.test_negative_samples_list = args.test_negative_samples_list
        self.cands_ids, self.index_ids = self._get_cands_and_index_ids()
        # self.create_dataset_partition_dict()

    def _get_cands_and_index_ids(self):
        dataset_config = json.load(open('dataset_configs.json'))[self.dataset_name]
        main_object_dict = getattr(self, f'_read_objects_{self.dataset_name}')(dataset_config)
        cands_ids = set(main_object_dict['cands'].keys())
        index_ids = set(main_object_dict['index'].keys())
        del main_object_dict
        return cands_ids, index_ids

    def create_dataset_partition_dict(self, seed):
        self.seed = seed
        cands_ids = self.cands_ids
        index_ids = self.index_ids
        train_negative_sampling_dict = self._get_train_negative_sampling_dict(cands_ids, index_ids)
        test_dict = self._get_test_ids_dict(cands_ids, index_ids, train_negative_sampling_dict)
        dataset_partition_dict = {'train': {'negative_sampling': train_negative_sampling_dict}, 'test': test_dict}
        self._save_dataset_partition_dict(dataset_partition_dict)

    def _get_train_negative_sampling_dict(self, cands_ids, index_ids):
        np.random.seed(self.seed)
        train_ids_dict = {}
        intersection_set = cands_ids.intersection(index_ids)
        for train_size, ratio_val in self.train_size_ratio_list.items():
            print(f"Creating training data for ({train_size})")
            train_ids_dict[train_size] = {}
            train_ids_size_num = int(ratio_val * len(intersection_set))
            train_ids_for_curr_size = set(np.random.choice(list(intersection_set), train_ids_size_num, replace=False))
            for neg_samples_num in self.train_neg_samples_list:
                train_ids_dict[train_size][neg_samples_num] = self._get_pairs_per_neg_samples(train_ids_for_curr_size,
                                                                                              index_ids,
                                                                                              neg_samples_num)
        return train_ids_dict

    def _get_pairs_per_neg_samples(self, ids_for_curr_size, index_ids, neg_samples_num):
        np.random.seed(self.seed)
        pos_pairs = [(cand_id, cand_id) for cand_id in ids_for_curr_size]
        neg_pairs = []
        for cand_id in ids_for_curr_size:
            neg_samples = set(np.random.choice(list(index_ids), neg_samples_num, replace=False))
            neg_pairs.extend([(cand_id, neg_sample) for neg_sample in neg_samples if neg_sample != cand_id])
        all_pairs = pos_pairs + neg_pairs
        np.random.shuffle(all_pairs)
        return all_pairs

    def _get_test_ids_dict(self, cands_ids, index_ids, train_ids_dict):
        test_ids_dict = {}
        intersection_set = cands_ids.intersection(index_ids)
        test_ids_dict['matching'] = self._get_test_pairs_for_matching(cands_ids, index_ids,
                                                                      intersection_set, train_ids_dict)
        test_ids_dict['blocking'] = self._get_test_data_for_blocking(index_ids, intersection_set, train_ids_dict)
        return test_ids_dict

    def _get_test_pairs_for_matching(self, cands_ids, index_ids, intersection_set, train_ids_dict):
        print("Creating test data for matching")
        test_matching_dict = {}
        test_matching_dict['negative_sampling'] = self._get_negative_sampling_test_ids_dict(index_ids,
                                                                                            intersection_set,
                                                                                            train_ids_dict)
        return test_matching_dict

    def _get_negative_sampling_test_ids_dict(self, index_ids, intersection_set, train_ids_dict):
        np.random.seed(self.seed)
        local_test_ids_dict = {}
        for test_size, ratio_val in self.test_size_ratio_list.items():
            print(f"Creating test data for matching ({test_size})")
            local_test_ids_dict[test_size] = {}
            corresponding_train_cands_ids = set \
                ([pair[0] for pair in train_ids_dict[test_size][self.train_neg_samples_list[0]]])
            potential_test_ids = intersection_set - corresponding_train_cands_ids
            test_ids_size_num = int(ratio_val * len(potential_test_ids))
            test_ids_for_curr_size = set(np.random.choice(list(potential_test_ids), test_ids_size_num, replace=False))
            for test_neg_samples in self.test_negative_samples_list:
                local_test_ids_dict[test_size][test_neg_samples] = self._get_pairs_per_neg_samples \
                    (test_ids_for_curr_size, index_ids, test_neg_samples)
        return local_test_ids_dict

    def _get_test_data_for_blocking(self, index_ids, intersection_set, train_ids_dict):
        test_blocking_dict = defaultdict(dict)
        for test_size, ratio_val in self.test_size_ratio_list.items():
            print(f"Creating test data for blocking ({test_size})")
            corresponding_train_cands_ids = set([pair[0] for pair in
                                                 train_ids_dict[test_size][self.train_neg_samples_list[0]]])
            potential_cands_test_ids = intersection_set - corresponding_train_cands_ids
            cands_test_ids = set(np.random.choice(list(potential_cands_test_ids),
                                                  int(ratio_val * len(potential_cands_test_ids)), replace=False))
            index_test_ids = cands_test_ids.copy()
            index_test_ids.update(set(np.random.choice(list(index_ids),
                                                       int(ratio_val * len(index_ids)),
                                                       replace=False)))
            test_blocking_dict[test_size] = {'cands': cands_test_ids, 'index': index_test_ids}
        return test_blocking_dict

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

    @staticmethod
    def _remove_train_objects_from_object_dict(object_dict, train_ids):
        for objects_type in object_dict.keys():
            object_dict[objects_type] = {
                object_id: object_data
                for object_id, object_data in object_dict[objects_type].items()
                if object_id not in train_ids
            }
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
                        new_obj_key = self.standardize_obj_key(obj_key, objects_type)
                        polygon_mesh_data = self._get_polygon_mesh(data, obj_key, vertices)
                        if polygon_mesh_data is not None:
                            object_dict[objects_type][new_obj_key] = polygon_mesh_data
                    except:
                        continue
        intersection_keys = set(object_dict['cands'].keys()).intersection(set(object_dict['index'].keys()))
        object_dict['cands'] = {obj_key: object_dict['cands'][obj_key] for obj_key in intersection_keys}
        print(f"Number of overlapping objects: {len(intersection_keys)}")
        print(f"Number of cands: {len(object_dict['cands'])}")
        print(f"Number of index: {len(object_dict['index'])}")
        for objects_type in objects_path_dict.keys():
            object_dict['mapping_dict'][objects_type] = {ind: obj_key for ind, obj_key in
                                                         enumerate(object_dict[objects_type].keys())}
            object_dict['inv_mapping_dict'][objects_type] = {obj_key: ind for ind, obj_key in
                                                             enumerate(object_dict[objects_type].keys())}
        return object_dict

    @staticmethod
    def standardize_obj_key(obj_key, object_type):
        if object_type == 'cands':
            return obj_key.split('bag_')[1]
        elif object_type == 'index':
            return obj_key.split('NL.IMBAG.Pand.')[1].split('-0')[0]
        else:
            raise ValueError('Invalid source')

    def _insert_polygon_mesh(self, object_dict, obj_type, obj_data, obj_ind=None):
        vertices = obj_data['vertices']
        obj_key = list(obj_data['CityObjects'].keys())[0]
        polygon_mesh = self._get_polygon_mesh(obj_data, obj_key, vertices)
        if polygon_mesh is not None:
            object_dict[obj_type][obj_ind] = polygon_mesh
        return object_dict

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

    @staticmethod
    def _compute_object_centroid(vertices):
        unique_vertices = np.array(vertices)
        return unique_vertices.mean(axis=0)

    @staticmethod
    def _get_vertices(polygon_mesh):
        return np.unique(np.array([coord for surface in polygon_mesh for coord in surface]), axis=0)

    def _save_dataset_partition_dict(self, dataset_partition_dict):
        if not os.path.exists(config.FilePaths.dataset_partition_path):
            os.makedirs(config.FilePaths.dataset_partition_path)
        path = f"{config.FilePaths.dataset_partition_path}{self.dataset_name}_seed{self.seed}.pkl"
        pkl.dump(dataset_partition_dict, open(path, 'wb'))
        print(f"Saved the dataset partition dict to {path}")
        return


def generate_partition_dicts(args):
    partition_dict_obj = DataPartitionGenerator(args)
    for seed in range(1, args.seeds_num + 1):
        start_time = time()
        partition_dict_obj.create_dataset_partition_dict(seed)
        end_time = time()
        print(f"Elapsed time for seed {seed}: {end_time - start_time}")
        print(3 * '--------------------------')
    print("Done!")


def get_potnetial_neg_pairs(dataset_size_version, bkafi_dim, train_or_test, seed):
    file_name = get_file_name()
    file_name.replace('concatenation', 'division')
    if train_or_test == 'train':
        file_name = file_name.replace('Operator', 'Train_Operator')
    blocking_results_dir = config.FilePaths.results_path + 'blocking_output/'
    blocking_results_path = (f"{blocking_results_dir}{file_name}_"
                             f"{dataset_size_version}_neg_samples_num2_vector_normalization_True_sdr_factor_False_"
                             f"bkafi_criterion=feature_importance_seed={seed}.joblib")
    print(f"\nLoading blocking results from {blocking_results_path}")
    blocking_dict = joblib.load(blocking_results_path)
    print(f"Loaded blocking results from {blocking_results_path}")
    neg_pairs = blocking_dict['neg_pairs'][bkafi_dim]
    return neg_pairs


def process_blocking_based_pairs(seed, neg_samples_num, cands_with_match_ids, potential_neg_pairs):
    np.random.seed(seed)
    pos_pairs = [(cand_id, cand_id) for cand_id in cands_with_match_ids]
    neg_pairs = potential_neg_pairs[neg_samples_num + 1]
    all_pairs = pos_pairs + neg_pairs
    np.random.shuffle(all_pairs)
    return all_pairs


def get_blocking_based_pairs(args, seed, train_or_test, dataset_partition_dict):
    local_test_ids_dict = {}
    neg_samples_list = args.train_neg_samples_list if train_or_test == 'train' else args.test_negative_samples_list
    for set_size in ['small', 'medium', 'large']:
        local_test_ids_dict[set_size] = {}
        if train_or_test == 'train':
            negative_sampling_pair_set = dataset_partition_dict['train']['negative_sampling'][set_size][2]
        else:
            negative_sampling_pair_set = dataset_partition_dict['test']['matching']['negative_sampling'][set_size][2]
        cands_with_match_ids = set([pair[0] for pair in negative_sampling_pair_set if pair[0] == pair[1]])
        potential_neg_pairs = get_potnetial_neg_pairs(set_size, args.bkafi_dim, train_or_test, seed)
        for neg_samples_num in neg_samples_list:
            local_test_ids_dict[set_size][neg_samples_num] = process_blocking_based_pairs(seed, neg_samples_num,
                                                                                          cands_with_match_ids,
                                                                                          potential_neg_pairs)
    return local_test_ids_dict


def add_blocking_based_mode_pairs(args):
    for seed in range(1, args.seeds_num + 1):
        # read the existing dataset partition dict

        dataset_partition_dict = pkl.load(open(f"data/dataset_partitions/{args.dataset_name}_seed{seed}.pkl", 'rb'))
        print(f"Loaded dataset partition dict for seed {seed} with blocking-based pairs")
        dataset_partition_dict['train']['blocking-based'] = get_blocking_based_pairs(args, seed, 'train',
                                                                                     dataset_partition_dict)
        dataset_partition_dict['test']['matching']['blocking-based'] = get_blocking_based_pairs(args, seed, 'test',
                                                                                                dataset_partition_dict)
        # save the updated dataset partition dict
        pkl.dump(dataset_partition_dict, open(f"data/dataset_partitions/{args.dataset_name}_seed{seed}.pkl", 'wb'))
        print(f"Updated dataset partition dict for seed {seed} with blocking-based pairs")
        print(3 * '--------------------------')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=config.Constants.dataset_name)
    parser.add_argument('--blocking_based_mode', type=bool, default=True)
    parser.add_argument('--seeds_num', type=int, default=config.Constants.seeds_num)
    parser.add_argument('--train_neg_samples_list', type=list, default=[2, 5])
    parser.add_argument('--test_negative_samples_list', type=list, default=[2, 5])
    parser.add_argument('--train_size_ratio_list', type=dict, default={"small": 0.1, "medium": 0.4, "large": 0.6})
    parser.add_argument('--test_size_ratio_list', type=dict, default={"small": 0.1, "medium": 0.5, "large": 1.0})
    parser.add_argument('--bkafi_dim', type=int, default=3)

    args = parser.parse_args()

    if args.blocking_based_mode:  # load existing partitions and add blocking-based partitions for the matching mode
        add_blocking_based_mode_pairs(args)
    else:
        generate_partition_dicts(args)
