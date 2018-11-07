from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import scipy.special
import matplotlib.pyplot
import mpl_toolkits.mplot3d

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, "../.."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import util.integrate
import util.mesh


def triangle_quadrature_standard_points_test1():
	max_quadrature_degree = 20
	max_function_degree = 30
	num_function_samples = 100
	abs_error_tensor = np.zeros((max_quadrature_degree + 1, max_function_degree + 1, num_function_samples))

	for quadrature_degree in range(max_quadrature_degree + 1):
		(point_matrix, weight_vector) = util.integrate.triangle_quadrature_standard_points(quadrature_degree)

		assert np.issubdtype(point_matrix.dtype, np.floating)
		assert (point_matrix.shape[1] == 2)

		assert np.issubdtype(weight_vector.dtype, np.floating)
		assert (weight_vector.ndim == 1)
		assert (point_matrix.shape[0] == weight_vector.size)
		num_points = point_matrix.shape[0]

		assert (np.abs(np.sum(weight_vector, axis=None, dtype=np.float64, keepdims=False) - 1.0) < 1e-6)

		xi_vector = np.reshape(point_matrix[:, 0], (1, 1, num_points))
		eta_vector = np.reshape(point_matrix[:, 1], (1, 1, num_points))

		for function_degree in range(max_function_degree + 1):
			(degree1_grid, degree2_grid) = np.meshgrid(
				np.arange(0, function_degree + 1, 1, dtype=np.int32), 
				np.arange(0, function_degree + 1, 1, dtype=np.int32), 
				indexing="ij", sparse=False, copy=True
			)
			coeff_valid_mask = ((degree1_grid + degree2_grid) <= function_degree)
			
			function_exponent_vector = np.linspace(0.0, float(function_degree), num=(function_degree + 1), endpoint=True, dtype=np.float64)
			xi_power_matrix = np.power(xi_vector, np.reshape(function_exponent_vector, (function_degree + 1, 1, 1)))
			eta_power_matrix = np.power(eta_vector, np.reshape(function_exponent_vector, (1, function_degree + 1, 1)))
			

			for function_index in range(num_function_samples):
				coeff_matrix = np.random.random([function_degree + 1, function_degree + 1]).astype(np.float64)
				coeff_matrix = coeff_matrix * coeff_valid_mask

				analytic_integral_matrix = np.exp(
					scipy.special.gammaln((degree1_grid + 1).astype(np.float64)) + 
					scipy.special.gammaln((degree2_grid + 1).astype(np.float64)) - 
					scipy.special.gammaln((degree1_grid + degree2_grid + 3).astype(np.float64))
				)
				analytic_result = np.sum(
					coeff_matrix * analytic_integral_matrix, 
					axis=None, dtype=np.float64, keepdims=False
				)

				quadrature_result = 0.5 * np.sum(
					(
						xi_power_matrix * np.reshape(coeff_matrix, (function_degree + 1, function_degree + 1, 1)) * eta_power_matrix * 
						np.reshape(weight_vector, (1, 1, num_points))
					), 
					axis=None, dtype=np.float64, keepdims=False
				)

				abs_error_tensor[quadrature_degree, function_degree, function_index] = np.abs(analytic_result - quadrature_result)
	
	mean_abs_error_matrix = np.mean(abs_error_tensor, axis=2, dtype=np.float64, keepdims=False)
	minimum_nonzero_value = np.amin(mean_abs_error_matrix[mean_abs_error_matrix > 0.0], axis=None, keepdims=False)
	log_mean_abs_error_matrix = np.log10(mean_abs_error_matrix + (mean_abs_error_matrix == 0.0) * minimum_nonzero_value)

	figure = matplotlib.pyplot.figure()
	axes = figure.add_subplot(1, 1, 1, projection="3d")

	(x_position_grid, y_position_grid) = np.meshgrid(
		np.linspace(0.0, float(max_quadrature_degree), num=(max_quadrature_degree + 1), endpoint=True, dtype=np.float64) - 0.5, 
		np.linspace(0.0, float(max_function_degree), num=(max_function_degree + 1), endpoint=True, dtype=np.float64) - 0.5, 
		indexing="ij", sparse=False, copy=True
	)
	x_position_vector = x_position_grid.flatten()
	y_position_vector = y_position_grid.flatten()

	axes.bar3d(
		x_position_vector, 
		y_position_vector, 
		np.amin(log_mean_abs_error_matrix, axis=None, keepdims=False) * np.ones_like(x_position_vector), 
		1.0, 
		1.0, 
		(log_mean_abs_error_matrix - np.amin(log_mean_abs_error_matrix, axis=None, keepdims=False)).flatten()
	)

	axes.set_xlabel("Quadrature Degree")
	axes.set_ylabel("Function Degree")
	axes.set_zlabel("log10(Absolute Error)")

	matplotlib.pyplot.show()


def triangle_quadrature_evaluation_points_test1():
	test_mesh = util.mesh.generate_icosahedron_mesh()
	test_triangle_tensor = test_mesh.vertices[test_mesh.faces, :]

	num_quadrature_points = 100
	num_triangles = test_triangle_tensor.shape[0]
	num_dimensions = test_triangle_tensor.shape[2]
	
	test_quadrature_point_matrix = np.random.random([num_quadrature_points, 2]) - 0.25
	test_evaluation_point_tensor = util.integrate.triangle_quadrature_evaluation_points(test_triangle_tensor, test_quadrature_point_matrix)

	gt_evaluation_point_tensor = np.zeros((num_triangles, num_quadrature_points, num_dimensions), dtype=np.float64)
	for triangle_index in range(num_triangles):
		for quadrature_point_index in range(num_quadrature_points):
			norm_coordinate_0 = test_quadrature_point_matrix[quadrature_point_index, 0]
			norm_coordinate_1 = test_quadrature_point_matrix[quadrature_point_index, 1]
			norm_coordinate_2 = (1.0 - norm_coordinate_0 - norm_coordinate_1)
			
			gt_evaluation_point_tensor[triangle_index, quadrature_point_index, :] = (
				norm_coordinate_0 * test_triangle_tensor[triangle_index, 0, :] + 
				norm_coordinate_1 * test_triangle_tensor[triangle_index, 1, :] + 
				norm_coordinate_2 * test_triangle_tensor[triangle_index, 2, :]
			)
	
	assert (test_evaluation_point_tensor.shape == gt_evaluation_point_tensor.shape)
	assert np.allclose(test_evaluation_point_tensor, gt_evaluation_point_tensor, rtol=1e-4, atol=1e-4, equal_nan=False)