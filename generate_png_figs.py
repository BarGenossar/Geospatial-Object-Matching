from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm
import argparse
from utils import *
import json


device = "cuda" if torch.cuda.is_available() else "cpu"

def find_min_coord(dim, polygon_mesh):
    return min([vertex[dim] for surface in polygon_mesh for vertex in surface])

def find_max_coord(dim, polygon_mesh):
    return max([vertex[dim] for surface in polygon_mesh for vertex in surface])


def plot_object(obj_id, polygon_mesh, save_dir, margin=3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Add polygon mesh
    for surface in polygon_mesh:
        poly = Poly3DCollection([surface], alpha=0.5, edgecolor='k')
        ax.add_collection3d(poly)

    # Set limits
    x_min = find_min_coord(0, polygon_mesh) - margin
    y_min = find_min_coord(1, polygon_mesh) - margin
    z_min = find_min_coord(2, polygon_mesh) - margin
    x_max = find_max_coord(0, polygon_mesh) + margin
    y_max = find_max_coord(1, polygon_mesh) + margin
    z_max = find_max_coord(2, polygon_mesh) + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.grid(False)  # Disable grid
    ax.set_axis_off()  # Remove the background planes
    plt.savefig(os.path.join(save_dir, f'{obj_id}.png'), format='png', bbox_inches='tight', pad_inches=0,
                transparent=True)
    # plt.show()
    plt.close(fig)


def get_object_dict_paths(dataset_name, dataset_size_version, evaluation_mode,
                          neg_samples_num, matching_cands_generation, seed):
    object_dict_path = f"{config.FilePaths.object_dict_path}{dataset_name}/"
    if not os.path.exists(object_dict_path):
        os.makedirs(object_dict_path)
    if evaluation_mode == 'blocking':
        train_full_path = f"{object_dict_path}train_blocking_{dataset_size_version}"
        test_full_path = f"{object_dict_path}test_blocking_{dataset_size_version}"
    else:
        train_full_path = (f"{object_dict_path}train_matching_{dataset_size_version}_"
                           f"neg_samples_num={neg_samples_num}")
        test_full_path = f"{object_dict_path}test_matching_{matching_cands_generation}" \
                         f"_{dataset_size_version}_neg_samples_num={neg_samples_num}"
    return f"{train_full_path}_seed_{seed}.joblib", f"{test_full_path}_seed_{seed}.joblib"


def plot_object_wrapper(args):
    obj_id, polygon_mesh, save_dir = args
    return plot_object(obj_id, polygon_mesh, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=config.Constants.dataset_name)
    parser.add_argument('--dataset_size_version', type=str, default=config.Constants.dataset_size_version)
    parser.add_argument('--evaluation_mode', type=str, default=config.Constants.evaluation_mode)
    parser.add_argument('--neg_samples_num', type=int, default=2)
    parser.add_argument('--matching_cands_generation', type=str, default=config.Constants.matching_cands_generation)
    parser.add_argument('--seeds_num', type=int, default=config.Constants.seeds_num)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    dataset_size_version = args.dataset_size_version
    evaluation_mode = args.evaluation_mode
    neg_samples_num = args.neg_samples_num
    matching_cands_generation = args.matching_cands_generation
    seeds_num = args.seeds_num

    dataset_config = json.load(open('dataset_configs.json'))[args.dataset_name]
    png_path_cands = f"{dataset_config['cands_path']}png_figs"
    png_path_index = f"{dataset_config['index_path']}png_figs"
    if not os.path.exists(png_path_cands):
        os.makedirs(png_path_cands)
    if not os.path.exists(png_path_index):
        os.makedirs(png_path_index)

    # for seed in range(1, seeds_num + 1):
    #     print(f"Seed: {seed}")
    #     train_object_dict_path, test_object_dict_path = get_object_dict_paths(dataset_name, dataset_size_version,
    #                                                                           evaluation_mode, neg_samples_num,
    #                                                                           matching_cands_generation, seed)
    #
    #     for object_dict_path in [train_object_dict_path, test_object_dict_path]:
    #
    #         object_dict = joblib.load(object_dict_path)
    #         for file_type, png_path in zip(['cands', 'index'], [png_path_cands, png_path_index]):
    #             print(f"Generating images for {file_type} in {object_dict_path}...")
    #             for obj_id, obj_data in tqdm.tqdm(object_dict[file_type].items(), mininterval=10.0):
    #                 polygon_mesh = obj_data['polygon_mesh']
    #                 plot_object(obj_id, polygon_mesh, png_path)
    # print("Done!")
    for seed in range(1, seeds_num + 1):
        print(f"Seed: {seed}")
        train_object_dict_path, test_object_dict_path = get_object_dict_paths(
            dataset_name,
            dataset_size_version,
            evaluation_mode,
            neg_samples_num,
            matching_cands_generation,
            seed
        )

        for object_dict_path in [train_object_dict_path, test_object_dict_path]:
            object_dict = joblib.load(object_dict_path)

            for file_type, png_path in zip(['cands', 'index'], [png_path_cands, png_path_index]):
                print(f"Generating images for {file_type} in {object_dict_path}...")

                args_list = [
                    (obj_id, obj_data['polygon_mesh'], png_path)
                    for obj_id, obj_data in object_dict[file_type].items()
                ]

                with Pool(cpu_count()-2) as pool:
                    list(tqdm.tqdm(
                        pool.imap_unordered(plot_object_wrapper, args_list),
                        total=len(args_list),
                        mininterval=10.0
                    ))

    print("Done!")