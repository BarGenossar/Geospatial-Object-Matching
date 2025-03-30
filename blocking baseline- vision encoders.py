import torch
import clip
from PIL import Image
import torchvision.transforms as transforms
import pickle
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import faiss
import numpy as np
import tqdm
import argparse
from utils import *


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vit_model', type=str, default="ViT-B/32")
    parser.add_argument('--generate_images', type=bool, default=True)

    args = parser.parse_args()
    if args.generate_images:
        train_object_dict_path = "data/RawCitiesData/The Hague/train_object_dict.joblib"
        test_object_dict_path = "data/RawCitiesData/The Hague/test_object_dict.joblib"
        # open a dir "png_figs" to save the images for each one of the sources A and B
        if not os.path.exists("data/RawCitiesData/The Hague/Source A/png_figs"):
            os.makedirs("data/RawCitiesData/The Hague/Source A/png_figs")
        if not os.path.exists("data/RawCitiesData/The Hague/Source B/png_figs"):
            os.makedirs("data/RawCitiesData/The Hague/Source B/png_figs")

        train_object_dict, test_object_dict = load_object_dict()

        for object_dict_path in [train_object_dict_path, test_object_dict_path]:
            object_dict = joblib.load(object_dict_path)
            for file_type in ['cands', 'index']:
                if file_type == 'index' and object_dict_path == test_object_dict_path:
                    continue
                source = 'A' if file_type == 'cands' else 'B'
                img_path = f"data/RawCitiesData/The Hague/Source {source}/png_figs"
                for obj_id, obj_data in tqdm.tqdm(object_dict[file_type].items()):
                    polygon_mesh = obj_data['polygon_mesh']
                    plot_object(obj_id, polygon_mesh, img_path)
    print("Done!")