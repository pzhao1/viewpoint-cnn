from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import scipy.special

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, "../.."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import util.geometry
import util.mesh
import util.sph_harm
import util.transform


def sh_linear_index_test1():
	assert (util.sph_harm.sh_degree_to_linear_index_range(0) == (0, 1))
	assert (util.sph_harm.sh_degree_to_linear_index_range(1) == (1, 4))
	assert (util.sph_harm.sh_degree_to_linear_index_range(2) == (4, 9))
	assert (util.sph_harm.sh_degree_to_linear_index_range(3) == (9, 16))
	assert (util.sph_harm.sh_degree_to_linear_index_range(10) == (100, 121))
	(linear_index_start_vector, linear_index_end_vector) = util.sph_harm.sh_degree_to_linear_index_range(
		[0, 1, 2, 3, 10]
	)
	assert np.all(np.equal(linear_index_start_vector, np.array([0, 1, 4, 9, 100], dtype=np.int32)), axis=None)
	assert np.all(np.equal(linear_index_end_vector, np.array([1, 4, 9, 16, 121], dtype=np.int32)), axis=None)

	assert (util.sph_harm.sh_degree_and_order_to_linear_index(0, 0) == 0)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(1, -1) == 1)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(1, 0) == 2)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(1, 1) == 3)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(2, -2) == 4)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(2, -1) == 5)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(2, 0) == 6)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(2, 1) == 7)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(2, 2) == 8)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(3, -3) == 9)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(3, -2) == 10)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(3, -1) == 11)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(3, 0) == 12)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(3, 1) == 13)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(3, 2) == 14)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(3, 3) == 15)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(10, -10) == 100)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(10, 0) == 110)
	assert (util.sph_harm.sh_degree_and_order_to_linear_index(10, 10) == 120)
	linear_index_vector = util.sph_harm.sh_degree_and_order_to_linear_index(
		[0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 10, 10, 10], 
		[0, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, -10, 0, 10]
	)
	assert np.all(np.equal(
		linear_index_vector, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 100, 110, 120], dtype=np.int32)
	), axis=None)

	assert (util.sph_harm.sh_linear_index_to_degree_and_order(0) == (0, 0))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(1) == (1, -1))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(2) == (1, 0))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(3) == (1, 1))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(4) == (2, -2))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(5) == (2, -1))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(6) == (2, 0))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(7) == (2, 1))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(8) == (2, 2))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(9) == (3, -3))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(10) == (3, -2))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(11) == (3, -1))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(12) == (3, 0))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(13) == (3, 1))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(14) == (3, 2))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(15) == (3, 3))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(100) == (10, -10))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(110) == (10, 0))
	assert (util.sph_harm.sh_linear_index_to_degree_and_order(120) == (10, 10))
	(weight_degree_array, weight_order_array) = util.sph_harm.sh_linear_index_to_degree_and_order(
		np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 100, 110, 120], dtype=np.int32)
	)
	assert (np.all(weight_degree_array == np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 10, 10, 10]), axis=None))
	assert (np.all(weight_order_array == np.array([0, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, -10, 0, 10]), axis=None))


def sh_evaluate_test1():
	azimuth_vector = np.linspace(0.0, 2.0 * np.pi, num=10, endpoint=False, dtype=np.float64)
	colatitude_vector = np.linspace(1e-4, np.pi-1e-4, num=10, endpoint=True, dtype=np.float64)
	(azimuth_matrix, colatitude_matrix) = np.meshgrid(azimuth_vector, colatitude_vector, indexing="ij", sparse=False, copy=True)

	for degree in range(5):
		for order in range(-degree, degree + 1):
			if (order == 0):
				scale_factor = 1.0
			else:
				scale_factor = np.sqrt(2.0)
			
			normalizing_factor = np.sqrt(
				((2.0 * degree + 1.0) * scipy.special.gamma(degree - np.abs(order) + 1.0)) / 
				((4.0 * np.pi) * scipy.special.gamma(degree + np.abs(order) + 1.0))
			)

			# Manually compute Legendre polynomial based on http://mathworld.wolfram.com/AssociatedLegendrePolynomial.html
			cos_colatitude_matrix = np.cos(colatitude_matrix)
			if (degree == 0):
				legendre_matrix = np.ones(cos_colatitude_matrix.shape, dtype=np.float64)
			elif (degree == 1):
				if (order == 0):
					legendre_matrix = np.copy(cos_colatitude_matrix)
				else:
					legendre_matrix = -1.0 * np.sqrt(1.0 - np.square(cos_colatitude_matrix))
			elif (degree == 2):
				if (order == 0):
					legendre_matrix = 0.5 * ((3.0 * np.square(cos_colatitude_matrix)) - 1.0)
				elif (abs(order) == 1):
					legendre_matrix = -3.0 * cos_colatitude_matrix * np.sqrt(1.0 - np.square(cos_colatitude_matrix))
				else:
					legendre_matrix = 3.0 * (1.0 - np.square(cos_colatitude_matrix))
			elif (degree == 3):
				if (order == 0):
					legendre_matrix = 0.5 * cos_colatitude_matrix * ((5.0 * np.square(cos_colatitude_matrix)) - 3.0)
				elif (abs(order) == 1):
					legendre_matrix = 1.5 * (1.0 - (5.0 * np.square(cos_colatitude_matrix))) * np.sqrt(1.0 - np.square(cos_colatitude_matrix))
				elif (abs(order) == 2):
					legendre_matrix = 15.0 * cos_colatitude_matrix * (1.0 - np.square(cos_colatitude_matrix))
				else:
					legendre_matrix = -15.0 * np.power(1.0 - np.square(cos_colatitude_matrix), 1.5)
			else:
				if (order == 0):
					legendre_matrix = 0.125 * ((35.0 * np.power(cos_colatitude_matrix, 4.0)) - (30.0 * np.square(cos_colatitude_matrix)) + 3.0)
				elif (abs(order) == 1):
					legendre_matrix = 2.5 * cos_colatitude_matrix * (3.0 - (7.0 * np.square(cos_colatitude_matrix))) * np.sqrt(1.0 - np.square(cos_colatitude_matrix))
				elif (abs(order) == 2):
					legendre_matrix = 7.5 * ((7.0 * np.square(cos_colatitude_matrix)) - 1.0) * (1.0 - np.square(cos_colatitude_matrix))
				elif (abs(order) == 3):
					legendre_matrix = -105.0 * cos_colatitude_matrix * np.power(1.0 - np.square(cos_colatitude_matrix), 1.5)
				else:
					legendre_matrix = 105.0 * np.square(1.0 - np.square(cos_colatitude_matrix))
			
			if (order > 0):
				azimuth_factor_matrix = np.cos(order * azimuth_matrix)
			elif (order == 0):
				azimuth_factor_matrix = np.ones(azimuth_matrix.shape, dtype=np.float64)
			else:
				azimuth_factor_matrix = np.sin(order * azimuth_matrix)

			true_spherical_harmonic_matrix = scale_factor * normalizing_factor * legendre_matrix * azimuth_factor_matrix
			computed_spherical_harmonic_matrix = util.sph_harm.sh_evaluate(degree, order, azimuth_matrix, colatitude_matrix)

			assert np.allclose(true_spherical_harmonic_matrix, computed_spherical_harmonic_matrix, rtol=1e-4, atol=1e-4, equal_nan=False)


def sh_evaluate_test2():
	sample_direction_mesh = util.mesh.generate_subdivided_icosahedron_mesh(4)
	sample_azimuth_vector = util.geometry.compute_points_azimuth(sample_direction_mesh.vertices)
	sample_colatitude_vector = util.geometry.compute_points_colatitude(sample_direction_mesh.vertices)

	spherical_harmonic_degree = 4
	spherical_harnomic_order = 3
	spherical_harmonic_value_vector = util.sph_harm.sh_evaluate(
		spherical_harmonic_degree, spherical_harnomic_order, sample_azimuth_vector, sample_colatitude_vector
	)
	
	util.mesh.visualize_spherical_function(sample_direction_mesh, spherical_harmonic_value_vector)


def sh_mixture_test1():
	sample_direction_mesh = util.mesh.generate_subdivided_icosahedron_mesh(4)
	sample_azimuth_vector = util.geometry.compute_points_azimuth(sample_direction_mesh.vertices)
	sample_colatitude_vector = util.geometry.compute_points_colatitude(sample_direction_mesh.vertices)

	sh_mixture = util.sph_harm.SHMixture()
	sh_mixture.set_all_weights([
		               0.0, 
		          0.0, 1.0, 0.0, 
		     0.0, 0.0, 0.0, 0.0, 0.0, 
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
	])
	sh_mixture_value_vector = sh_mixture.evaluate(sample_azimuth_vector, sample_colatitude_vector)

	util.mesh.visualize_spherical_function(sample_direction_mesh, sh_mixture_value_vector)


def sh_mixture_test2():
	sample_direction_mesh = util.mesh.generate_subdivided_icosahedron_mesh(4)
	sample_azimuth_vector = util.geometry.compute_points_azimuth(sample_direction_mesh.vertices)
	sample_colatitude_vector = util.geometry.compute_points_colatitude(sample_direction_mesh.vertices)

	sh_mixture = util.sph_harm.SHMixture()
	sh_mixture.set_degree_weights(1, [0.0, 1.0, 0.0])
	sh_mixture.set_degree_weights(3, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
	sh_mixture_value_vector = sh_mixture.evaluate(sample_azimuth_vector, sample_colatitude_vector)

	util.mesh.visualize_spherical_function(sample_direction_mesh, sh_mixture_value_vector)


def sh_mixture_test3():
	sample_direction_mesh = util.mesh.generate_subdivided_icosahedron_mesh(4)
	sample_azimuth_vector = util.geometry.compute_points_azimuth(sample_direction_mesh.vertices)
	sample_colatitude_vector = util.geometry.compute_points_colatitude(sample_direction_mesh.vertices)

	sh_mixture = util.sph_harm.SHMixture()
	sh_mixture.set_single_weight(1, 0, 1.0)
	sh_mixture.set_single_weight(3, 0, 1.0)
	sh_mixture_value_vector = sh_mixture.evaluate(sample_azimuth_vector, sample_colatitude_vector)

	util.mesh.visualize_spherical_function(sample_direction_mesh, sh_mixture_value_vector)


def sh_mixture_test4():
	# Test symmetry detection basic usage.
	sh_mixture = util.sph_harm.SHMixture()
	sh_mixture.set_all_weights([
		               0.1, 
		          0.0, 0.3, 0.0, 
		     0.0, 0.0, 0.5, 0.0, 0.0, 
		0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0
	])
	(is_zonal, discrete_symmetry_fold) = sh_mixture.detect_z_rotation_symmetry(1e-4, 1e-4)
	assert is_zonal

	sh_mixture.set_all_weights([
		               10.0, 
		          0.1, 10.0, 0.0, 
		     0.0, 0.0, 10.0, 0.0, 0.0, 
		0.0, 0.0, 0.0, 10.0, 0.0, 0.1, 0.0
	])
	(is_zonal, discrete_symmetry_fold) = sh_mixture.detect_z_rotation_symmetry(0.1, 0.01)
	assert is_zonal

	sh_mixture.set_all_weights([
		               10.0, 
		          0.1, 10.0, 0.0, 
		     0.0, 0.0, 10.0, 0.0, 0.0, 
		0.0, 0.0, 0.0, 10.0, 0.0, 0.1, 0.0
	])
	(is_zonal, discrete_symmetry_fold) = sh_mixture.detect_z_rotation_symmetry(0.09, 0.01)
	assert (not is_zonal)
	assert (discrete_symmetry_fold == 1)

	sh_mixture.set_all_weights([
		               10.0, 
		          0.1, 10.0, 0.0, 
		     0.0, 0.0, 10.0, 0.0, 0.0, 
		0.0, 0.0, 0.0, 10.0, 0.0, 0.1, 0.0
	])
	(is_zonal, discrete_symmetry_fold) = sh_mixture.detect_z_rotation_symmetry(0.1, 0.009)
	assert (not is_zonal)
	assert (discrete_symmetry_fold == 1)

	# Test symmetry detection special cases - extremely large or small weights.
	sh_mixture.set_all_weights([
		                0.0 , 
		          1e12, 0.0 , 0.0, 
		     0.0, 0.0 , 1e14, 0.0, 0.0, 
		0.0, 0.0, 0.0 , 0.0 , 0.0, 0.0, 0.0
	])
	(is_zonal, discrete_symmetry_fold) = sh_mixture.detect_z_rotation_symmetry(np.inf, 0.01)
	assert is_zonal

	sh_mixture.set_all_weights([
		                0.0, 
		          1e-9, 0.0, 0.0, 
		     0.0, 0.0 , 0.0, 0.0, 0.0, 
		0.0, 0.0, 0.0 , 0.0, 0.0, 0.0, 0.0
	])
	(is_zonal, discrete_symmetry_fold) = sh_mixture.detect_z_rotation_symmetry(np.inf, 0.01)
	assert (not is_zonal)
	assert (discrete_symmetry_fold == 1)

	sh_mixture.set_all_weights([
		                   0.0 , 
		            1e-20, 0.0 , 0.0, 
		       0.0, 0.0  , 1e20, 0.0, 0.0, 
		1e-20, 0.0, 0.0  , 0.0 , 0.0, 0.0, 0.0
	])
	(is_zonal, discrete_symmetry_fold) = sh_mixture.detect_z_rotation_symmetry(0.0, 0.01)
	assert (not is_zonal)
	assert (discrete_symmetry_fold == 1)

	# Test discrete symmetry angle.
	sh_mixture.set_all_weights([
		                              0.0, 
		                         0.0, 0.0, 0.0, 
		                    0.0, 0.0, 1.0, 0.0, 0.0, 
		               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 
		     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	])
	(is_zonal, discrete_symmetry_fold) = sh_mixture.detect_z_rotation_symmetry(1e-4, 1e-4)
	assert (not is_zonal)
	assert (discrete_symmetry_fold == 2)

	sh_mixture.set_all_weights([
		                              0.0, 
		                         0.0, 0.0, 0.0, 
		                    0.0, 0.0, 1.0, 0.0, 0.0, 
		               0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 
		          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 
		     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	])
	(is_zonal, discrete_symmetry_fold) = sh_mixture.detect_z_rotation_symmetry(1e-4, 1e-4)
	assert (not is_zonal)
	assert (discrete_symmetry_fold == 1)

	sh_mixture.set_all_weights([
		                              0.0, 
		                         0.0, 0.0, 0.0, 
		                    0.0, 0.0, 1.0, 0.0, 0.0, 
		               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	])
	(is_zonal, discrete_symmetry_fold) = sh_mixture.detect_z_rotation_symmetry(1e-4, 1e-4)
	assert (not is_zonal)
	assert (discrete_symmetry_fold == 6)

	sh_mixture.set_all_weights([
		                              0.0, 
		                         0.0, 0.0, 0.0, 
		                    0.0, 0.0, 1.0, 0.0, 0.0, 
		               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 
		0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	])
	(is_zonal, discrete_symmetry_fold) = sh_mixture.detect_z_rotation_symmetry(1e-4, 1e-4)
	assert (not is_zonal)
	assert (discrete_symmetry_fold == 2)

	sh_mixture.set_all_weights([
		                              0.0, 
		                         0.0, 0.0, 0.0, 
		                    0.0, 0.0, 1.0, 0.0, 0.0, 
		               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 
		     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	])
	(is_zonal, discrete_symmetry_fold) = sh_mixture.detect_z_rotation_symmetry(1e-4, 1e-4)
	assert (not is_zonal)
	assert (discrete_symmetry_fold == 3)

	sh_mixture.set_all_weights([
		                              0.0, 
		                         0.0, 0.0, 0.0, 
		                    0.0, 0.0, 1.0, 0.0, 0.0, 
		               0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 
		          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	])
	(is_zonal, discrete_symmetry_fold) = sh_mixture.detect_z_rotation_symmetry(1e-4, 1e-4)
	assert (not is_zonal)
	assert (discrete_symmetry_fold == 2)

	sh_mixture.set_all_weights([
		                              0.0, 
		                         0.0, 0.0, 0.0, 
		                    0.0, 0.0, 1.0, 0.0, 0.0, 
		               0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 
		          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
		0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	])
	(is_zonal, discrete_symmetry_fold) = sh_mixture.detect_z_rotation_symmetry(1e-4, 1e-4)
	assert (not is_zonal)
	assert (discrete_symmetry_fold == 1)


def sh_rotation_transform_test1():
	sample_direction_mesh = util.mesh.generate_subdivided_icosahedron_mesh(4)
	sample_azimuth_vector = util.geometry.compute_points_azimuth(sample_direction_mesh.vertices)
	sample_colatitude_vector = util.geometry.compute_points_colatitude(sample_direction_mesh.vertices)

	sh_mixture = util.sph_harm.SHMixture()
	sh_mixture.set_all_weights([
		               0.0, 
		          0.0, 1.0, 0.0, 
		     0.0, 0.0, 0.0, 0.0, 0.0, 
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
	])
	sh_mixture_value_vector = sh_mixture.evaluate(sample_azimuth_vector, sample_colatitude_vector)

	util.mesh.visualize_spherical_function(sample_direction_mesh, sh_mixture_value_vector)

	rotation_matrix = util.transform.axis_angle_to_rotation_matrix(np.array([0.0, 1.0, 0.0], dtype=np.float64), np.pi/2)
	sh_mixture.apply_rotation(rotation_matrix)
	sh_mixture_value_vector = sh_mixture.evaluate(sample_azimuth_vector, sample_colatitude_vector)

	util.mesh.visualize_spherical_function(sample_direction_mesh, sh_mixture_value_vector)


def sh_rotation_transform_test2():
	sample_direction_mesh = util.mesh.generate_subdivided_icosahedron_mesh(4)
	sample_azimuth_vector = util.geometry.compute_points_azimuth(sample_direction_mesh.vertices)
	sample_colatitude_vector = util.geometry.compute_points_colatitude(sample_direction_mesh.vertices)

	sh_mixture = util.sph_harm.SHMixture()
	sh_mixture.set_all_weights([
		               0.0, 
		          0.0, 0.0, 1.0, 
		     0.0, 0.0, 1.0, 0.0, 0.0, 
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
	])
	sh_mixture_value_vector = sh_mixture.evaluate(sample_azimuth_vector, sample_colatitude_vector)

	util.mesh.visualize_spherical_function(sample_direction_mesh, sh_mixture_value_vector)

	rotation_matrix = util.transform.axis_angle_to_rotation_matrix(np.array([1.0, 0.0, 1.0], dtype=np.float64), np.pi)
	sh_mixture.apply_rotation(rotation_matrix)
	sh_mixture_value_vector = sh_mixture.evaluate(sample_azimuth_vector, sample_colatitude_vector)

	util.mesh.visualize_spherical_function(sample_direction_mesh, sh_mixture_value_vector)


def sh_rotation_transform_test3():
	sample_direction_mesh = util.mesh.generate_subdivided_icosahedron_mesh(4)
	sample_azimuth_vector = util.geometry.compute_points_azimuth(sample_direction_mesh.vertices)
	sample_colatitude_vector = util.geometry.compute_points_colatitude(sample_direction_mesh.vertices)

	sh_mixture = util.sph_harm.SHMixture()
	sh_mixture.set_all_weights([
		               0.0, 
		          1.0, 0.0, 0.0, 
		     0.0, 0.0, 1.0, 0.0, 0.0, 
		0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
	])
	sh_mixture_value_vector = sh_mixture.evaluate(sample_azimuth_vector, sample_colatitude_vector)

	util.mesh.visualize_spherical_function(sample_direction_mesh, sh_mixture_value_vector)

	rotation_matrix = util.transform.axis_angle_to_rotation_matrix(np.array([1.0, 1.0, 1.0], dtype=np.float64), np.pi/2)
	sh_mixture.apply_rotation(rotation_matrix)
	sh_mixture_value_vector = sh_mixture.evaluate(sample_azimuth_vector, sample_colatitude_vector)

	util.mesh.visualize_spherical_function(sample_direction_mesh, sh_mixture_value_vector)


def sh_rotation_transform_test4():
	sample_direction_mesh = util.mesh.generate_subdivided_icosahedron_mesh(4)
	sample_azimuth_vector = util.geometry.compute_points_azimuth(sample_direction_mesh.vertices)
	sample_colatitude_vector = util.geometry.compute_points_colatitude(sample_direction_mesh.vertices)

	sh_mixture = util.sph_harm.SHMixture()
	sh_mixture.set_all_weights(np.random.random([49]))
	sh_mixture_value_vector = sh_mixture.evaluate(sample_azimuth_vector, sample_colatitude_vector)

	util.mesh.visualize_spherical_function(sample_direction_mesh, sh_mixture_value_vector)

	rotation_matrix = util.transform.axis_angle_to_rotation_matrix(np.array([0.0, 1.0, 0.0], dtype=np.float64), np.pi)
	sh_mixture.apply_rotation(rotation_matrix)
	sh_mixture_value_vector = sh_mixture.evaluate(sample_azimuth_vector, sample_colatitude_vector)

	util.mesh.visualize_spherical_function(sample_direction_mesh, sh_mixture_value_vector)


def sh_rotation_transform_test5():
	rotation_matrix = util.transform.axis_angle_to_rotation_matrix(np.array([1.0, 1.0, 1.0], dtype=np.float64), np.pi)

	sh_mixture = util.sph_harm.SHMixture()
	sh_mixture.set_all_weights(np.random.random([81]))
	rotated_sh_mixture = sh_mixture.copy()
	rotated_sh_mixture.apply_rotation(rotation_matrix)

	sample_direction_mesh = util.mesh.generate_subdivided_icosahedron_mesh(5)
	inv_transform_matrix = np.eye(4, 4, dtype=np.float64)
	inv_transform_matrix[0:3, 0:3] = np.transpose(rotation_matrix)
	inv_rotated_sample_direction_mesh = sample_direction_mesh.copy().apply_transform(inv_transform_matrix)

	sample_azimuth_vector = util.geometry.compute_points_azimuth(sample_direction_mesh.vertices)
	sample_colatitude_vector = util.geometry.compute_points_colatitude(sample_direction_mesh.vertices)
	rotated_sh_mixture_value_vector = rotated_sh_mixture.evaluate(sample_azimuth_vector, sample_colatitude_vector)

	inv_rotated_sample_azimuth_vector = util.geometry.compute_points_azimuth(inv_rotated_sample_direction_mesh.vertices)
	inv_rotated_sample_colatitude_vector = util.geometry.compute_points_colatitude(inv_rotated_sample_direction_mesh.vertices)
	inv_rotated_direction_value_vector = sh_mixture.evaluate(inv_rotated_sample_azimuth_vector, inv_rotated_sample_colatitude_vector)

	assert np.allclose(rotated_sh_mixture_value_vector, inv_rotated_direction_value_vector, rtol=1e-4, atol=1e-4, equal_nan=False)


def sh_zonal_rotation_transform_test1():
	max_degree = 20
	(_, num_components) = util.sph_harm.sh_degree_to_linear_index_range(max_degree)

	rotation_matrix = util.transform.axis_angle_to_rotation_matrix(np.random.random([3]), np.random.random() * np.pi)
	
	azimuth = util.geometry.compute_points_azimuth(rotation_matrix[:, 2])
	colatitude = util.geometry.compute_points_colatitude(rotation_matrix[:, 2])

	zonal_transform_vector = util.sph_harm.sh_zonal_rotation_transform(azimuth, colatitude, max_degree)
	assert (zonal_transform_vector.shape == (num_components, ))

	all_transform_matrix = util.sph_harm.sh_rotation_transform(rotation_matrix, max_degree).toarray()

	for degree in range(max_degree + 1):
		for order in range(-degree, degree + 1):
			map_to_linear_index = util.sph_harm.sh_degree_and_order_to_linear_index(degree, order)
			map_from_linear_index = util.sph_harm.sh_degree_and_order_to_linear_index(degree, 0)

			assert (np.abs(
				all_transform_matrix[map_to_linear_index, map_from_linear_index] - 
				zonal_transform_vector[map_to_linear_index])
			< 1e-4)


def sh_zonal_rotation_transform_test2():
	num_samples = 6
	max_degree = 20
	(_, num_components) = util.sph_harm.sh_degree_to_linear_index_range(max_degree)

	rotation_tensor = np.zeros((num_samples, ) + (3, 3), dtype=np.float64)
	for sample_index in range(num_samples):
		rotation_tensor[sample_index, :, :] = util.transform.axis_angle_to_rotation_matrix(np.random.random([3]), np.random.random() * np.pi)
	
	azimuth_vector = util.geometry.compute_points_azimuth(rotation_tensor[:, :, 2])
	colatitude_vector = util.geometry.compute_points_colatitude(rotation_tensor[:, :, 2])

	zonal_transform_matrix = util.sph_harm.sh_zonal_rotation_transform(azimuth_vector, colatitude_vector, max_degree)
	assert (zonal_transform_matrix.shape == (num_samples, ) + (num_components, ))

	for sample_index in range(num_samples):
		all_transform_matrix = util.sph_harm.sh_rotation_transform(rotation_tensor[sample_index, :, :], max_degree).toarray()

		for degree in range(max_degree + 1):
			for order in range(-degree, degree + 1):
				map_to_linear_index = util.sph_harm.sh_degree_and_order_to_linear_index(degree, order)
				map_from_linear_index = util.sph_harm.sh_degree_and_order_to_linear_index(degree, 0)

				assert (np.abs(
					all_transform_matrix[map_to_linear_index, map_from_linear_index] - 
					zonal_transform_matrix[sample_index, map_to_linear_index])
				< 1e-4)
		
		print ("Completed testing ", sample_index, " out of ", num_samples)


def sh_zonal_rotation_transform_test3():
	sample_shape = (2, 3)
	max_degree = 20
	(_, num_components) = util.sph_harm.sh_degree_to_linear_index_range(max_degree)

	rotation_tensor = np.zeros(sample_shape + (3, 3), dtype=np.float64)
	for sample_index1 in range(sample_shape[0]):
		for sample_index2 in range(sample_shape[1]):
			rotation_tensor[sample_index1, sample_index2, :, :] = util.transform.axis_angle_to_rotation_matrix(np.random.random([3]), np.random.random() * np.pi)
	
	azimuth_matrix = util.geometry.compute_points_azimuth(rotation_tensor[:, :, :, 2])
	colatitude_matrix = util.geometry.compute_points_colatitude(rotation_tensor[:, :, :, 2])

	zonal_transform_tensor = util.sph_harm.sh_zonal_rotation_transform(azimuth_matrix, colatitude_matrix, max_degree)
	assert (zonal_transform_tensor.shape == (sample_shape) + (num_components, ))

	for sample_index1 in range(sample_shape[0]):
		for sample_index2 in range(sample_shape[1]):
			all_transform_matrix = util.sph_harm.sh_rotation_transform(rotation_tensor[sample_index1, sample_index2, :, :], max_degree).toarray()

			for degree in range(max_degree + 1):
				for order in range(-degree, degree + 1):
					map_to_linear_index = util.sph_harm.sh_degree_and_order_to_linear_index(degree, order)
					map_from_linear_index = util.sph_harm.sh_degree_and_order_to_linear_index(degree, 0)

					assert (np.abs(
						all_transform_matrix[map_to_linear_index, map_from_linear_index] - 
						zonal_transform_tensor[sample_index1, sample_index2, map_to_linear_index])
					< 1e-4)
			
			print ("Completed testing ", (sample_index1, sample_index2), " out of ", sample_shape)


def sh_zonal_rotation_transform_test4():
	sample_shape = (1000, 79)
	max_degree = 8
	(_, num_components) = util.sph_harm.sh_degree_to_linear_index_range(max_degree)

	rotation_tensor = np.zeros(sample_shape + (3, 3), dtype=np.float64)
	for sample_index1 in range(sample_shape[0]):
		for sample_index2 in range(sample_shape[1]):
			rotation_tensor[sample_index1, sample_index2, :, :] = util.transform.axis_angle_to_rotation_matrix(np.random.random([3]), np.random.random() * np.pi)
	
	azimuth_matrix = util.geometry.compute_points_azimuth(rotation_tensor[:, :, :, 2])
	colatitude_matrix = util.geometry.compute_points_colatitude(rotation_tensor[:, :, :, 2])

	start_time_sec = time.time()
	zonal_transform_tensor = util.sph_harm.sh_zonal_rotation_transform(azimuth_matrix, colatitude_matrix, max_degree)
	end_time_sec = time.time()

	print("Calculating zonal rotation transform for %d samples with degree %d took %f seconds" % (
		np.prod(sample_shape, axis=None, dtype=np.int32, keepdims=False), 
		max_degree, 
		(end_time_sec - start_time_sec)
	))

	assert (zonal_transform_tensor.shape == (sample_shape) + (num_components, ))