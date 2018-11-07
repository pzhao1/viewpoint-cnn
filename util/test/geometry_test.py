from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import scipy.spatial

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, "../.."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import util.geometry
import util.mesh


def compute_points_distance_test1():
	num_points1 = 100
	num_points2 = 200
	num_dimensions = 1000
	
	point_matrix1 = np.random.random([num_points1, num_dimensions])
	point_matrix2 = np.random.random([num_points2, num_dimensions])
	gt_distance_matrix = scipy.spatial.distance.cdist(point_matrix1, point_matrix2, metric="euclidean")

	test_distance_matrix = util.geometry.compute_points_distance(point_matrix1, point_matrix2)

	assert np.all(np.abs(test_distance_matrix - gt_distance_matrix) < 1e-4, axis=None, keepdims=False)


def compute_points_distance_test2():
	num_points1 = 1000
	num_points2 = 1000
	num_dimensions = 10000

	point_matrix1 = np.random.random([num_points1, num_dimensions])
	point_matrix2 = np.random.random([num_points2, num_dimensions])

	start_sec = time.time()
	test_distance_matrix = util.geometry.compute_points_distance(point_matrix1, point_matrix2, verbose=True)
	end_sec = time.time()

	print("Computing distance between (%d, %d) points of %d dimensions took my function %f seconds" % (num_points1, num_points2, num_dimensions, end_sec - start_sec))

	start_sec = time.time()
	gt_distance_matrix = scipy.spatial.distance.cdist(point_matrix1, point_matrix2, metric="euclidean")
	end_sec = time.time()

	assert np.all(np.abs(test_distance_matrix - gt_distance_matrix) < 1e-4, axis=None, keepdims=False)
	print("Computing distance between (%d, %d) points of %d dimensions took Scipy %f seconds" % (num_points1, num_points2, num_dimensions, end_sec - start_sec))


def compute_points_pairwise_distance_test1():
	num_points = 100
	num_dimensions = 1000

	point_matrix = np.random.random([num_points, num_dimensions])
	gt_distance_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(point_matrix, metric="euclidean"))

	test_distance_matrix = util.geometry.compute_points_pairwise_distance(point_matrix) 

	assert np.all(np.abs(test_distance_matrix - gt_distance_matrix) < 1e-4, axis=None, keepdims=False)


def compute_points_pairwise_distance_test2():
	num_points = 1000
	num_dimensions = 10000

	point_matrix = np.random.random([num_points, num_dimensions])

	start_sec = time.time()
	test_distance_matrix = util.geometry.compute_points_pairwise_distance(point_matrix, verbose=True)
	end_sec = time.time()

	print("Computing pairwise distance for %d points of %d dimensions took my function %f seconds" % (num_points, num_dimensions, end_sec - start_sec))

	start_sec = time.time()
	gt_distance_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(point_matrix, metric="euclidean"))
	end_sec = time.time()

	assert np.all(np.abs(test_distance_matrix - gt_distance_matrix) < 1e-4, axis=None, keepdims=False)
	print("Computing pairwise distance for %d points of %d dimensions took Scipy %f seconds" % (num_points, num_dimensions, end_sec - start_sec))


def compute_triangles_area_test1():
	test_mesh = util.mesh.generate_icosahedron_mesh()
	normalized_test_mesh_vertices = test_mesh.vertices / np.linalg.norm(test_mesh.vertices, ord=2, axis=1, keepdims=True)
	test_triangle_vertex_tensor = normalized_test_mesh_vertices[test_mesh.faces, :]

	test_triangle_area_vector = util.geometry.compute_triangles_area(test_triangle_vertex_tensor)

	# From https://en.wikipedia.org/wiki/Regular_icosahedron#Cartesian_coordinates, 
	# the area of each triangle should be the area of an equilateral triangle with size 2 scaled by
	# the inverse square of vertex norms.
	gt_triangle_area_vector = (
		np.sqrt(3) * np.ones((test_triangle_vertex_tensor.shape[0], ), dtype=np.float64) / 
		(np.square(1.0) + np.square((1.0 + np.sqrt(5.0)) / 2.0))
	)

	assert (test_triangle_area_vector.shape == gt_triangle_area_vector.shape)
	assert np.allclose(test_triangle_area_vector, gt_triangle_area_vector, rtol=1e-4, atol=1e-4, equal_nan=False)


def rectangle_shape_test1():
	num_samples = 100

	for sample_index in range(num_samples):
		gt_size_x = np.random.random()
		gt_size_y = np.random.random()
		test_rectangle_size = util.geometry.RectangleShape(gt_size_x, gt_size_y)
		assert (abs(test_rectangle_size.size_x - gt_size_x) < 1e-4)
		assert (abs(test_rectangle_size.size_y - gt_size_y) < 1e-4)

		gt_size_x = np.random.random()
		gt_size_y = np.random.random()
		test_rectangle_size.size_x = gt_size_x
		test_rectangle_size.size_y = gt_size_y
		assert (abs(test_rectangle_size.size_x - gt_size_x) < 1e-4)
		assert (abs(test_rectangle_size.size_y - gt_size_y) < 1e-4)

		test_rectangle_size2 = test_rectangle_size.copy()
		assert (abs(test_rectangle_size2.size_x - gt_size_x) < 1e-4)
		assert (abs(test_rectangle_size2.size_y - gt_size_y) < 1e-4)

		gt_size_x2 = np.random.random()
		gt_size_y2 = np.random.random()
		test_rectangle_size2.size_x = gt_size_x2
		test_rectangle_size2.size_y = gt_size_y2
		assert (abs(test_rectangle_size2.size_x - gt_size_x2) < 1e-4)
		assert (abs(test_rectangle_size2.size_y - gt_size_y2) < 1e-4)

		gt_scale_factor = 2.0 * np.random.random()
		gt_size_x2 = gt_scale_factor * gt_size_x2
		gt_size_y2 = gt_scale_factor * gt_size_y2
		test_rectangle_size2.apply_scale(gt_scale_factor)
		assert (abs(test_rectangle_size2.size_x - gt_size_x2) < 1e-4)
		assert (abs(test_rectangle_size2.size_y - gt_size_y2) < 1e-4)

		assert (abs(test_rectangle_size.size_x - gt_size_x) < 1e-4)
		assert (abs(test_rectangle_size.size_y - gt_size_y) < 1e-4)


def rectangle_test1():
	num_samples = 100

	def check_rectangle_properties(test_rectangle, gt_min_x, gt_min_y, gt_size_x, gt_size_y):
		assert (abs(test_rectangle.min_x - gt_min_x) < 1e-4)
		assert (abs(test_rectangle.min_y - gt_min_y) < 1e-4)
		assert (abs(test_rectangle.size_x - gt_size_x) < 1e-4)
		assert (abs(test_rectangle.size_y - gt_size_y) < 1e-4)
		assert (abs(test_rectangle.max_size - max(gt_size_x, gt_size_y)) < 1e-4)
		assert (abs(test_rectangle.min_size - min(gt_size_x, gt_size_y)) < 1e-4)
		assert (abs(test_rectangle.max_x - (gt_min_x + gt_size_x)) < 1e-4)
		assert (abs(test_rectangle.max_y - (gt_min_y + gt_size_y)) < 1e-4)
		assert (abs(test_rectangle.center_x - (gt_min_x + (gt_size_x / 2.0))) < 1e-4)
		assert (abs(test_rectangle.center_y - (gt_min_y + (gt_size_y / 2.0))) < 1e-4)


	for sample_index in range(num_samples):
		gt_min_x = np.random.random()
		gt_min_y = np.random.random()
		gt_size_x = np.random.random()
		gt_size_y = np.random.random()
		test_rectangle = util.geometry.Rectangle(gt_min_x, gt_min_y, gt_size_x, gt_size_y)
		check_rectangle_properties(test_rectangle, gt_min_x, gt_min_y, gt_size_x, gt_size_y)

		gt_min_x = np.random.random()
		gt_min_y = np.random.random()
		gt_size_x = np.random.random()
		gt_size_y = np.random.random()
		test_rectangle.min_x = gt_min_x
		test_rectangle.min_y = gt_min_y
		test_rectangle.shape = util.geometry.RectangleShape(gt_size_x, gt_size_y)
		check_rectangle_properties(test_rectangle, gt_min_x, gt_min_y, gt_size_x, gt_size_y)

		gt_size_x = np.random.random()
		gt_size_y = np.random.random()
		test_rectangle.size_x = gt_size_x
		test_rectangle.size_y = gt_size_y
		check_rectangle_properties(test_rectangle, gt_min_x, gt_min_y, gt_size_x, gt_size_y)

		gt_min_x2 = gt_min_x
		gt_min_y2 = gt_min_y
		gt_size_x2 = gt_size_x
		gt_size_y2 = gt_size_y
		test_rectangle2 = test_rectangle.copy()
		check_rectangle_properties(test_rectangle2, gt_min_x2, gt_min_y2, gt_size_x2, gt_size_y2)

		gt_min_x2 = gt_min_x2 if (gt_size_x2 >= gt_size_y2) else (gt_min_x2 - ((gt_size_y2 - gt_size_x2) / 2.0))
		gt_min_y2 = gt_min_y2 if (gt_size_x2 < gt_size_y2) else (gt_min_y2 - ((gt_size_x2 - gt_size_y2) / 2.0))
		gt_size_x2 = max(gt_size_x2, gt_size_y2)
		gt_size_y2 = max(gt_size_x2, gt_size_y2)
		test_rectangle2.expand_to_square()
		check_rectangle_properties(test_rectangle2, gt_min_x2, gt_min_y2, gt_size_x2, gt_size_y2)

		gt_translation_x2 = np.random.random()
		gt_translation_y2 = np.random.random()
		gt_min_x2 = gt_min_x2 + gt_translation_x2
		gt_min_y2 = gt_min_y2 + gt_translation_y2
		test_rectangle2.apply_translation(gt_translation_x2, gt_translation_y2)
		check_rectangle_properties(test_rectangle2, gt_min_x2, gt_min_y2, gt_size_x2, gt_size_y2)

		gt_scale_factor = 2.0 * np.random.random()
		gt_size_x2 = gt_scale_factor * gt_size_x2
		gt_size_y2 = gt_scale_factor * gt_size_y2
		test_rectangle2.apply_scale_fixing_min_corner(gt_scale_factor)
		check_rectangle_properties(test_rectangle2, gt_min_x2, gt_min_y2, gt_size_x2, gt_size_y2)

		old_center_x2 = test_rectangle2.center_x
		old_center_y2 = test_rectangle2.center_y
		gt_scale_factor = 2.0 * np.random.random()
		test_rectangle2.apply_scale_fixing_centroid(gt_scale_factor)
		assert (abs(old_center_x2 - test_rectangle2.center_x) < 1e-4)
		assert (abs(old_center_y2 - test_rectangle2.center_y) < 1e-4)
		gt_min_x2 = test_rectangle2.min_x
		gt_min_y2 = test_rectangle2.min_y
		gt_size_x2 = gt_scale_factor * gt_size_x2
		gt_size_y2 = gt_scale_factor * gt_size_y2
		check_rectangle_properties(test_rectangle2, gt_min_x2, gt_min_y2, gt_size_x2, gt_size_y2)

		# Since rectangle2 is copied from rectangle, make sure all chcanges to rectangle2 did not affect rectangle.
		check_rectangle_properties(test_rectangle, gt_min_x, gt_min_y, gt_size_x, gt_size_y)

		test_rectangle3 = test_rectangle2.copy()
		assert test_rectangle2.contains_rectangle(test_rectangle3)
		assert (not test_rectangle2.strictly_contains_rectangle(test_rectangle3))

		gt_min_x3 = gt_min_x2
		gt_min_y3 = gt_min_y2
		gt_scale_factor = np.random.random()
		gt_size_x3 = gt_size_x2 * gt_scale_factor
		gt_size_y3 = gt_size_y2 * gt_scale_factor
		test_rectangle3.apply_scale_fixing_min_corner(gt_scale_factor)
		assert test_rectangle2.contains_rectangle(test_rectangle3)
		assert (not test_rectangle2.strictly_contains_rectangle(test_rectangle3))

		gt_translation_x3 = gt_size_x3 * (0.1 + 0.8 * np.random.random()) * (1.0 - gt_scale_factor)
		gt_translation_y3 = gt_size_y3 * (0.1 + 0.8 * np.random.random()) * (1.0 - gt_scale_factor)
		gt_min_x3 = gt_min_x3 + gt_translation_x3
		gt_min_y3 = gt_min_y3 + gt_translation_y3
		test_rectangle3.apply_translation(gt_translation_x3, gt_translation_y3)
		assert test_rectangle2.contains_rectangle(test_rectangle3)
		assert test_rectangle2.strictly_contains_rectangle(test_rectangle3)

		gt_translation_x3 = 2.0 * (gt_min_x2 - gt_min_x3)
		gt_translation_y3 = 2.0 * (gt_min_y2 - gt_min_y3)
		gt_min_x3 = gt_min_x3 + gt_translation_x3
		gt_min_y3 = gt_min_y3 + gt_translation_y3

		test_rectangle3.apply_translation(gt_translation_x3, gt_translation_y3)
		assert (not test_rectangle2.contains_rectangle(test_rectangle3))
		assert (not test_rectangle2.strictly_contains_rectangle(test_rectangle3))


def compute_points_azimuth_test1():
	num_samples = 100

	for sample_index in range(num_samples):
		gt_azimuth = 2.0 * np.pi * np.random.random()
		gt_colatitude = np.pi * np.random.random()
		gt_distance = np.random.random()
		
		test_point_array = np.array([
			gt_distance * np.sin(gt_colatitude) * np.cos(gt_azimuth), 
			gt_distance * np.sin(gt_colatitude) * np.sin(gt_azimuth), 
			gt_distance * np.cos(gt_colatitude)
		], dtype=np.float64)

		computed_azimuth = util.geometry.compute_points_azimuth(test_point_array)
		assert computed_azimuth.shape == ()
		assert (abs(gt_azimuth - np.mod(computed_azimuth, 2.0 * np.pi)) < 1e-4)


def compute_points_azimuth_test2():
	sample_shape = (20, 30, 40)

	gt_azimuth_array = 2.0 * np.pi * np.random.random(sample_shape)
	gt_colatitude_array = np.pi * np.random.random(sample_shape)
	gt_distance_array = np.random.random(sample_shape)

	test_point_array = np.stack([
		gt_distance_array * np.sin(gt_colatitude_array) * np.cos(gt_azimuth_array), 
		gt_distance_array * np.sin(gt_colatitude_array) * np.sin(gt_azimuth_array), 
		gt_distance_array * np.cos(gt_colatitude_array)
	], axis=len(sample_shape))

	computed_azimuth_array = util.geometry.compute_points_azimuth(test_point_array)
	assert computed_azimuth_array.shape == sample_shape
	assert np.allclose(gt_azimuth_array, np.mod(computed_azimuth_array, 2.0 * np.pi), rtol=1e-4, atol=1e-4, equal_nan=False)


def compute_points_colatitude_test1():
	num_samples = 100

	for sample_index in range(num_samples):
		gt_azimuth = 2.0 * np.pi * np.random.random()
		gt_colatitude = np.pi * np.random.random()
		gt_distance = np.random.random()
		
		test_point_array = np.array([
			gt_distance * np.sin(gt_colatitude) * np.cos(gt_azimuth), 
			gt_distance * np.sin(gt_colatitude) * np.sin(gt_azimuth), 
			gt_distance * np.cos(gt_colatitude)
		], dtype=np.float64)

		computed_colatitude = util.geometry.compute_points_colatitude(test_point_array)
		assert computed_colatitude.shape == ()
		assert (abs(gt_colatitude - computed_colatitude) < 1e-4)


def compute_points_colatitude_test2():
	sample_shape = (20, 30, 40)

	gt_azimuth_array = 2.0 * np.pi * np.random.random(sample_shape)
	gt_colatitude_array = np.pi * np.random.random(sample_shape)
	gt_distance_array = np.random.random(sample_shape)

	test_point_array = np.stack([
		gt_distance_array * np.sin(gt_colatitude_array) * np.cos(gt_azimuth_array), 
		gt_distance_array * np.sin(gt_colatitude_array) * np.sin(gt_azimuth_array), 
		gt_distance_array * np.cos(gt_colatitude_array)
	], axis=len(sample_shape))

	computed_colatitude_array = util.geometry.compute_points_colatitude(test_point_array)
	assert computed_colatitude_array.shape == sample_shape
	assert np.allclose(gt_colatitude_array, computed_colatitude_array, rtol=1e-4, atol=1e-4, equal_nan=False)


def compute_points_elevation_test1():
	num_samples = 100

	for sample_index in range(num_samples):
		gt_azimuth = 2.0 * np.pi * np.random.random()
		gt_elevation = (np.pi * np.random.random()) - (np.pi / 2.0)
		gt_distance = np.random.random()
		
		test_point_array = np.array([
			gt_distance * np.cos(gt_elevation) * np.cos(gt_azimuth), 
			gt_distance * np.cos(gt_elevation) * np.sin(gt_azimuth), 
			gt_distance * np.sin(gt_elevation)
		], dtype=np.float64)

		computed_elevation = util.geometry.compute_points_elevation(test_point_array)
		assert computed_elevation.shape == ()
		assert (abs(gt_elevation - computed_elevation) < 1e-4)


def compute_points_elevation_test2():
	sample_shape = (20, 30, 40)

	gt_azimuth_array = 2.0 * np.pi * np.random.random(sample_shape)
	gt_elevation_array = np.pi * np.random.random(sample_shape)  - (np.pi / 2.0)
	gt_distance_array = np.random.random(sample_shape)

	test_point_array = np.stack([
		gt_distance_array * np.cos(gt_elevation_array) * np.cos(gt_azimuth_array), 
		gt_distance_array * np.cos(gt_elevation_array) * np.sin(gt_azimuth_array), 
		gt_distance_array * np.sin(gt_elevation_array)
	], axis=len(sample_shape))

	computed_elevation_array = util.geometry.compute_points_elevation(test_point_array)
	assert computed_elevation_array.shape == sample_shape
	assert np.allclose(gt_elevation_array, computed_elevation_array, rtol=1e-4, atol=1e-4, equal_nan=False)