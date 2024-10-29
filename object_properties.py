from scipy.spatial import ConvexHull
import numpy as np
import faiss
import config
from shapely import Polygon, LineString
from scipy.spatial import KDTree
from collections import defaultdict
from utils import *
import math


class ObjectPropertiesProcessor:
    def __init__(self, object_dict):
        self.object_dict = object_dict
        # self.road_dict = self._get_road_dict()
        self.x_coordinates = self._get_specific_coordinates(0)
        self.y_coordinates = self._get_specific_coordinates(1)
        self.z_coordinates = self._get_specific_coordinates(2)
        self.centroids = self._create_centroids_dict()
        self.eigen_dict = self._create_eigen_dict()
        self.prop_names_dict = self._get_properties_dict()
        self._create_prop_vals_dict()

    # @staticmethod
    # def _get_road_dict():
    #     road_json = read_roads_from_json(config.FilePaths.roads_path)
    #     road_dict = defaultdict(dict)
    #     for road_ind, road in enumerate(road_json['features']):
    #         road_dict[road_ind] = convert_coords(road['geometry']['coordinates'])
    #     return road_dict

    def _get_specific_coordinates(self, coordinate_index):
        specidic_coord_dict = defaultdict(dict)
        for object_type, object_dict in self.object_dict.items():
            for file_ind, vertices in object_dict.items():
                specidic_coord_dict[object_type][file_ind] = [coord[coordinate_index] for coord in vertices]
        return specidic_coord_dict

    def _create_centroids_dict(self):
        centroids_dict = defaultdict(dict)
        for object_type, object_dict in self.object_dict.items():
            for file_ind, vertices in object_dict.items():
                centroids_dict[object_type][file_ind] = get_centroid(vertices)
        return centroids_dict

    def _create_prop_vals_dict(self):
        properties = config.Features.object_properties
        self.prop_vals_dict = dict()
        for prop in properties:
            self.prop_vals_dict[prop] = dict()
            for obj_type, object_dict in self.object_dict.items():
                self.prop_vals_dict[prop][obj_type] = dict()
                for file_ind, vertices in object_dict.items():
                    self.prop_vals_dict[prop][obj_type][file_ind] = self.prop_names_dict[prop](vertices, obj_type,
                                                                                               file_ind)
        return

    def _get_properties_dict(self):
        return {prop: getattr(self, f'_get_{prop}') for prop in config.Features.object_properties}

    # def _create_feature_vectors(self, feature):
    #     feature_vectors = []
    #     for pair in self.pairs_list:
    #         feature_vectors.append(self._get_feature_vec(pair))
    #     return feature_vectors

    def _get_bounding_box_width(self, vertices, object_type, file_ind):
        x_coords = self.x_coordinates[object_type][file_ind]
        return max(x_coords) - min(x_coords)

    def _get_bounding_box_length(self, vertices, object_type, file_ind):
        y_coords = self.y_coordinates[object_type][file_ind]
        return max(y_coords) - min(y_coords)

    def _get_aligned_bounding_box_width(self, vertices, object_type, file_ind):
        eigenvectors = self.eigen_dict[object_type][file_ind]['eigenvectors']
        aligned_vertices = np.dot(vertices, eigenvectors)
        min_vals = np.min(aligned_vertices, axis=0)
        max_vals = np.max(aligned_vertices, axis=0)
        return max_vals[0] - min_vals[0]

    def _get_aligned_bounding_box_length(self, vertices, object_type, file_ind):
        eigenvectors = self.eigen_dict[object_type][file_ind]['eigenvectors']
        aligned_vertices = np.dot(vertices, eigenvectors)
        min_vals = np.min(aligned_vertices, axis=0)
        max_vals = np.max(aligned_vertices, axis=0)
        return max_vals[1] - min_vals[1]

    def _get_aligned_bounding_box_height(self, vertices, object_type, file_ind):
        eigenvectors = self.eigen_dict[object_type][file_ind]['eigenvectors']
        aligned_vertices = np.dot(vertices, eigenvectors)
        min_vals = np.min(aligned_vertices, axis=0)
        max_vals = np.max(aligned_vertices, axis=0)
        return max_vals[2] - min_vals[2]

    def _get_area(self, vertices, object_type, file_ind):
        if file_ind not in self.prop_vals_dict['area'][object_type].keys():
            area = Polygon(vertices).area
            self.prop_vals_dict['area'][object_type][file_ind] = area
        else:
            area = self.prop_vals_dict['area'][object_type][file_ind]
        # Prevent cases where the area is 0
        return max(area, 0.1)

    def _get_perimeter(self, vertices, object_type, file_ind):
        if file_ind not in self.prop_vals_dict['perimeter'][object_type].keys():
            perimeter = Polygon(vertices).length
            self.prop_vals_dict['perimeter'][object_type][file_ind] = perimeter
        else:
            perimeter = self.prop_vals_dict['perimeter'][object_type][file_ind]
        return perimeter

    def _get_perimeter_index(self, vertices, object_type, file_ind):
        area = self._get_area(vertices, object_type, file_ind)
        perimeter = self._get_perimeter(vertices, object_type, file_ind)
        return 2 * math.sqrt(math.pi * area) /perimeter

    def _get_volume(self, vertices, object_type, file_ind):
        if file_ind not in self.prop_vals_dict['volume'][object_type].keys():
            area = self._get_area(vertices, object_type, file_ind)
            volume = self._get_height_diff(vertices, object_type, file_ind) * area
            self.prop_vals_dict['volume'][object_type][file_ind] = volume
        else:
            volume = self.prop_vals_dict['volume'][object_type][file_ind]
        return volume

    @staticmethod
    def _get_convex_hull_area(vertices, object_type, file_ind):
        return Polygon(vertices).convex_hull.area

    def _get_ave_centroid_distance(self, vertices, object_type, file_ind):
        centroid = self.centroids[object_type][file_ind]
        return np.mean([((coord[0] - centroid[0])**2 +
                         (coord[1] - centroid[1])**2 +
                         (coord[2] - centroid[2])**2)**0.5
                        for coord in vertices])

    def _get_max_height(self, vertices, object_type, file_ind):
        return max(self.z_coordinates[object_type][file_ind])

    def _get_min_height(self, vertices, object_type, file_ind):
        return min(self.z_coordinates[object_type][file_ind])

    def _get_height_diff(self, vertices, object_type, file_ind):
        return self._get_max_height(vertices, object_type, file_ind) - \
               self._get_min_height(vertices, object_type, file_ind)

    def _get_num_floors(self, vertices, object_type, file_ind):
        return len(set(self.z_coordinates[object_type][file_ind]))

    def _get_axes_symmetry(self, vertices, object_type, file_ind):
        x_coords = self.x_coordinates[object_type][file_ind]
        y_coords = self.y_coordinates[object_type][file_ind]
        z_coords = self.z_coordinates[object_type][file_ind]
        return np.mean([np.std(x_coords), np.std(y_coords), np.std(z_coords)])

    def _get_compactness(self, vertices, object_type, file_ind):
        area = self._get_area(vertices, object_type, file_ind)
        return self._get_convex_hull_area(vertices, object_type, file_ind) / area

    def _get_density(self, vertices, object_type, file_ind):
        area = self._get_area(vertices, object_type, file_ind)
        return area / self._get_perimeter(vertices, object_type, file_ind)

    def _create_eigen_dict(self):
        eigen_dict = dict()
        for obj_type, object_dict in self.object_dict.items():
            eigen_dict[obj_type] = dict()
            for file_ind, vertices in object_dict.items():
                eigen_dict[obj_type][file_ind] = dict()
                covariance_matrix = np.cov(vertices, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
                eigen_dict[obj_type][file_ind]['eigenvalues'] = eigenvalues
                eigen_dict[obj_type][file_ind]['eigenvectors'] = eigenvectors
        return eigen_dict

    def _get_elongation(self, vertices, object_type, file_ind):
        eigenvalues = self.eigen_dict[object_type][file_ind]['eigenvalues']
        return np.sqrt(eigenvalues.max() / eigenvalues.min())

    def _get_shape_index(self, vertices, object_type, file_ind):
        if self._get_area(vertices, object_type, file_ind) == 0:
            return 0
        return (self._get_perimeter(vertices, object_type, file_ind) /
                math.sqrt(4 * np.pi * self._get_area(vertices, object_type, file_ind)))

    def _get_hemisphericality(self, vertices, object_type, file_ind):
        area = self._get_area(vertices, object_type, file_ind)
        volume = self._get_volume(vertices, object_type, file_ind)
        return 3 * math.sqrt(2) * math.sqrt(math.pi) * volume / (math.pow(area, 1.5))

    def _get_fractality(self, vertices, object_type, file_ind):
        area = self._get_area(vertices, object_type, file_ind)
        volume = self._get_volume(vertices, object_type, file_ind)
        return 1 - math.log(volume) / (1.5 * math.log(area))

    def _get_cubeness(self, vertices, object_type, file_ind):
        area = self._get_area(vertices, object_type, file_ind)
        volume = self._get_volume(vertices, object_type, file_ind)
        return 6 * math.pow(volume, 2/3) / area

    def _get_circumference(self, vertices, object_type, file_ind):
        area = self._get_area(vertices, object_type, file_ind)
        volume = self._get_volume(vertices, object_type, file_ind)
        return 4 * math.pi * math.pow(3 * volume / (4 * math.pi), 2/3) / area

    def _get_num_vertices(self, vertices, object_type, file_ind):
        return len(vertices)

