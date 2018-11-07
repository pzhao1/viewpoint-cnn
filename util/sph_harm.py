from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import numpy as np
import scipy.sparse
import scipy.special

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, ".."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import util.transform


def sh_degree_to_linear_index_range(degree_array):
	if (not isinstance(degree_array, np.ndarray)):
		degree_array = np.array(degree_array, dtype=np.int32)
	assert np.issubdtype(degree_array.dtype, np.integer)
	assert np.all((degree_array >= 0), axis=None)

	return ((degree_array * degree_array), ((degree_array + 1) * (degree_array + 1)))


def sh_degree_and_order_to_linear_index(degree_array, order_array):
	if (not isinstance(degree_array, np.ndarray)):
		degree_array = np.array(degree_array, dtype=np.int32)
	assert np.issubdtype(degree_array.dtype, np.integer)
	assert np.all((degree_array >= 0), axis=None)

	if (not isinstance(order_array, np.ndarray)):
		order_array = np.array(order_array, dtype=np.int32)
	assert np.issubdtype(order_array.dtype, np.integer)
	assert np.all((np.abs(order_array) <= degree_array), axis=None)
	
	return (np.square(degree_array) + degree_array + order_array)


def sh_linear_index_to_degree_and_order(weight_index_array):
	if (not isinstance (weight_index_array, np.ndarray)):
		weight_index_array = np.array(weight_index_array, dtype=np.int32)
	assert np.issubdtype(weight_index_array.dtype, np.integer)
	assert np.all((weight_index_array >= 0), axis=None)

	weight_degree = np.floor(np.sqrt(weight_index_array.astype(np.float64))).astype(np.int32)
	weight_order = (weight_index_array - (np.square(weight_degree) + weight_degree)).astype(np.int32)

	return (weight_degree, weight_order)


def sh_evaluate(degree_array, order_array, azimuth_array, colatitude_array):
	"""
	Compute the values of real spherical harmonics given order, degree, azimuth, and colatitude.
	This is a element-wise function, which means all inputs' shapes must be the same, or compatible for broadcasting.
	"""
	# Check arguments.
	if (not isinstance(degree_array, np.ndarray)):
		degree_array = np.array(degree_array, dtype=np.int32)
	assert np.issubdtype(degree_array.dtype, np.integer)
	assert np.all((degree_array >= 0), axis=None)

	if (not isinstance(order_array, np.ndarray)):
		order_array = np.array(order_array, dtype=np.int32)
	assert np.issubdtype(order_array.dtype, np.integer)
	assert np.all((np.abs(order_array) <= degree_array), axis=None)
	
	if (not isinstance(azimuth_array, np.ndarray)):
		azimuth_array = np.array(azimuth_array, dtype=np.float64)
	assert np.issubdtype(azimuth_array.dtype, np.floating)
	azimuth_array = np.mod(azimuth_array, 2.0 * np.pi)
	
	if (not isinstance(colatitude_array, np.ndarray)):
		colatitude_array = np.array(colatitude_array, dtype=np.float64)
	assert np.issubdtype(colatitude_array.dtype, np.floating)
	assert np.all((colatitude_array >= 0.0), axis=None)
	assert np.all((colatitude_array <= np.pi), axis=None)

	# Compute complex spherical harmonics using SciPy
	complex_spherical_harmonic_array = scipy.special.sph_harm(order_array, degree_array, azimuth_array, colatitude_array)

	# Convert to real spherical harmonics
	order_positive_mask = (order_array > 0)
	order_negative_mask = (order_array < 0)
	order_zero_mask = (order_array == 0)

	real_spherical_harmonic_array = (order_zero_mask * np.real(complex_spherical_harmonic_array)) + np.sqrt(2) * (
		order_positive_mask * np.real(complex_spherical_harmonic_array) + 
		# When order is negative, scipy calculates legendre(order, degree), while the expression of real sperical
		# harmonics has legendre(-order, degree). There is a (-1)^order difference between the two.
		order_negative_mask * np.power(-1.0, order_array.astype(np.float64)) * np.imag(complex_spherical_harmonic_array)
	)

	return real_spherical_harmonic_array


def _sh_rotation_transform_P(i, L, a, b, transform_matrix):
	"""
	Compute term P in Table II of the reference paper.
	
	References: 
		Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion, Joseph Ivanic and Klaus Ruedenberg, 1996
		Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion (Additions and Corrections), Joseph Ivanic and Klaus Ruedenberg, 1998
	"""
	index_base_one = sh_degree_and_order_to_linear_index(1, 0)
	index_base_Lm1 = sh_degree_and_order_to_linear_index(L - 1, 0)

	ri1  = transform_matrix[index_base_one + i, index_base_one + 1]
	ri0  = transform_matrix[index_base_one + i, index_base_one    ]
	rim1 = transform_matrix[index_base_one + i, index_base_one - 1]

	if (b == L):
		return ((ri1 * transform_matrix[index_base_Lm1 + a, index_base_Lm1 + L - 1]) - (rim1 * transform_matrix[index_base_Lm1 + a, index_base_Lm1 - L + 1]))
	elif (b == -L):
		return ((ri1 * transform_matrix[index_base_Lm1 + a, index_base_Lm1 - L + 1]) + (rim1 * transform_matrix[index_base_Lm1 + a, index_base_Lm1 + L - 1]))
	else:
		return (ri0 * transform_matrix[index_base_Lm1 + a, index_base_Lm1 + b])

def _sh_rotation_transform_U(L, m, n, transform_matrix):
	"""
	Compute term U in Table II of the reference paper.
	
	References: 
		Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion, Joseph Ivanic and Klaus Ruedenberg, 1996
		Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion (Additions and Corrections), Joseph Ivanic and Klaus Ruedenberg, 1998
	"""
	return _sh_rotation_transform_P(0, L, m, n, transform_matrix)

def _sh_rotation_transform_V(L, m, n, transform_matrix):
	"""
	Compute term V in Table II of the reference paper.
	
	References:
		Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion, Joseph Ivanic and Klaus Ruedenberg, 1996
		Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion (Additions and Corrections), Joseph Ivanic and Klaus Ruedenberg, 1998
	"""
	if (m == 0):
		return (_sh_rotation_transform_P(1, L, 1, n, transform_matrix) + _sh_rotation_transform_P(-1, L, -1, n, transform_matrix))
	elif (m > 0):
		delta_m1 = 1.0 if (m == 1) else 0.0
		return (
			(_sh_rotation_transform_P(1, L, m - 1, n, transform_matrix) * np.sqrt(1.0 + delta_m1)) - 
			(_sh_rotation_transform_P(-1, L, -m + 1, n, transform_matrix) * (1.0 - delta_m1))
		)
	else:
		delta_mm1 = 1.0 if (m == -1) else 0.0
		return (
			(_sh_rotation_transform_P(1, L, m + 1, n, transform_matrix) * (1.0 - delta_mm1)) + 
			(_sh_rotation_transform_P(-1, L, -m - 1, n, transform_matrix) * np.sqrt(1.0 + delta_mm1))
		)

def _sh_rotation_transform_W(L, m, n, transform_matrix):
	"""
	Compute term W in Table II of the reference paper.
	
	References: 
		Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion, Joseph Ivanic and Klaus Ruedenberg, 1996
		Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion (Additions and Corrections), Joseph Ivanic and Klaus Ruedenberg, 1998
	"""
	assert (not (m == 0))

	if (m > 0):
		return (_sh_rotation_transform_P(1, L, m + 1, n, transform_matrix) + _sh_rotation_transform_P(-1, L, -m - 1, n, transform_matrix))
	else:
		return (_sh_rotation_transform_P(1, L, m - 1, n, transform_matrix) - _sh_rotation_transform_P(-1, L, -m + 1, n, transform_matrix))

def sh_rotation_transform(rotation_matrix, max_degree):
	"""
	Given a rotation matrix, calculate the transformation matrix for spherical harmonics coefficients
	induced by this rotation.
	
	References: 
		Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion, Joseph Ivanic and Klaus Ruedenberg, 1996
		Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion (Additions and Corrections), Joseph Ivanic and Klaus Ruedenberg, 1998
	"""
	assert util.transform.is_rotation_matrix(rotation_matrix)
	max_degree = int(max_degree)
	assert (max_degree >= 0)

	(_, num_components) = sh_degree_to_linear_index_range(max_degree)
	transform_dok_matrix = scipy.sparse.dok_matrix((num_components, num_components), dtype=np.float64)
	transform_dok_matrix[0, 0] = 1.0
	if (max_degree == 0):
		return scipy.sparse.csr_matrix(transform_dok_matrix)
	
	# Elements for degree = 1 are directly related to the rotation matrix.
	index_base_one = sh_degree_and_order_to_linear_index(1, 0)
	transform_dok_matrix[index_base_one - 1, index_base_one - 1] =  rotation_matrix[1, 1]
	transform_dok_matrix[index_base_one - 1, index_base_one    ] =  rotation_matrix[1, 2]
	transform_dok_matrix[index_base_one - 1, index_base_one + 1] = -rotation_matrix[1, 0]
	transform_dok_matrix[index_base_one    , index_base_one - 1] =  rotation_matrix[2, 1]
	transform_dok_matrix[index_base_one    , index_base_one    ] =  rotation_matrix[2, 2]
	transform_dok_matrix[index_base_one    , index_base_one + 1] = -rotation_matrix[2, 0]
	transform_dok_matrix[index_base_one + 1, index_base_one - 1] = -rotation_matrix[0, 1]
	transform_dok_matrix[index_base_one + 1, index_base_one    ] = -rotation_matrix[0, 2]
	transform_dok_matrix[index_base_one + 1, index_base_one + 1] =  rotation_matrix[0, 0]
	if (max_degree == 1):
		return transform_dok_matrix
	
	# Compute transformation for each subsequent degree recursively.
	# Notation follows the paper in reference, except use n in place of m', and use L in place of l (to distinguish from 1).
	for L in range(2, max_degree + 1):
		index_base_L = sh_degree_and_order_to_linear_index(L, 0)

		for m in range(-L, L + 1):
			delta_m0 = 1.0 if (m == 0) else 0.0

			for n in range(-L, L + 1):
				if (abs(n) == L):
					denominator = (2.0 * L) * (2.0 * L - 1)
				else:
					denominator = (L + n) * (L - n)
				
				# Compute u, v, and w in Table I in paper
				u = np.sqrt((L + m) * (L - m) / denominator)
				v = 0.5 * np.sqrt((1.0 + delta_m0) * (L + abs(m) - 1.0) * (L + abs(m)) / denominator) * (1.0 - (2.0 * delta_m0))
				w = (-0.5) * np.sqrt((L - abs(m) - 1.0) * (L - abs(m)) / denominator) * (1.0 - delta_m0)

				# Compute Eq 8.1 in paper
				r_Lmn = 0.0
				if (abs(u) > 1e-6):
					r_Lmn = r_Lmn + (u * _sh_rotation_transform_U(L, m, n, transform_dok_matrix))
				if (abs(v) > 1e-6):
					r_Lmn = r_Lmn + (v * _sh_rotation_transform_V(L, m, n, transform_dok_matrix))
				if (abs(w) > 1e-6):
					r_Lmn = r_Lmn + (w * _sh_rotation_transform_W(L, m, n, transform_dok_matrix))

				transform_dok_matrix[index_base_L + m, index_base_L + n] = r_Lmn
	
	return transform_dok_matrix


def sh_zonal_rotation_transform(azimuth_array, colatitude_array, max_degree):
	"""
	Given an array of rotations, calculate the transformation for the zonal spherical harmonics coefficients induced by them.

	This is MUCH faster than computing the entire transform matrix (using sh_rotation_transform) and taking the zonal componenets:
		1. The transformation can be computed directly, not recursively.
		2. This is a vectorized implementation that can compute transformations for many rotations at once.
	
	Notice the rotations are specified with 2 degrees of freedom only, since we only care about the azimuth and colatitude of the 
	rotated z-axis. The 3rd degree of freedom (axial rotation) does not affect the results, since we are rotating zonal spherical harmonics.
	If you have a rotation matrix, you can convert it into this format using:
		azimuth = compute_points_azimuth(rotation_matrix[:, 2]), or azimuth = np.arctan2(rotation_matrix[1, 2], rotation_matrix[0, 2])
		colatitude = compute_points_colatitude(rotation_matrix[:, 2]), or colatitude = np.arccos(rotation_matrix[2, 2] / norm(rotation_matrix[:, 2]))

	Inputs:
		azimuth_array   : An np.ndarray instance containing the azimuth of the rotated z-axis for each rotation.
		colatitude_array: An np.ndarray instance containign the colatitude of the rotated z-axis for each rotation.
		max_degree      : An integer containing the maximum degree of spherical harmonics to compute transformation for.
	
	Outputs:
		transform_array : An np.ndarray with dimension (input_shape + [num_components]) containing the transformation, 
		                  Transformation for input rotation at index (i, j, k, ...) can be accessed by transform_array[i, j, k, ..., :].
		                  The last dimension contains transformation factors arranged in linear index. See sh_linear_index_to_degree_and_order().
		                  Transformation factor from zonal harmonic of degree L to spherical harmonic of degree L and order M is stored in the 
		                  linear index corresponding to degree L and order M.
	
	References:
		http://research.spa.aalto.fi/projects/sht-lib/sht.html#6
		http://www.tau.ac.il/~tsirel/dump/Static/knowino.org/wiki/Wigner_D-matrix.html
	"""
	if (not isinstance(azimuth_array, np.ndarray)):
		azimuth_array = np.array(azimuth_array, dtype=np.float64)
	assert np.issubdtype(azimuth_array.dtype, np.floating)
	num_input_dimensions = azimuth_array.ndim

	if (not isinstance(colatitude_array, np.ndarray)):
		colatitude_array = np.array(colatitude_array, dtype=np.float64)
	assert np.issubdtype(colatitude_array.dtype, np.floating)
	assert (colatitude_array.ndim == num_input_dimensions)
	
	max_degree = int(max_degree)
	assert (max_degree >= 0)

	(_, num_components) = sh_degree_to_linear_index_range(max_degree)
	component_index_vector = np.arange(0, num_components, 1, dtype=np.int32)
	(component_degree_vector, component_order_vector) = sh_linear_index_to_degree_and_order(component_index_vector)

	# Based on https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html, broadcasting starts from trailing dimenisions.
	# This means if the dimensions azimuth_array and colatitude_array are expanded by 1 at the end, any broadcasting-compatible
	# operation will broadcast component_degree_vector and component_order_vector across the newly expanded dimension.
	complex_sh_value_array = scipy.special.sph_harm(
		component_order_vector, 
		component_degree_vector, 
		np.expand_dims(azimuth_array, num_input_dimensions), 
		np.expand_dims(colatitude_array, num_input_dimensions)
	)
	complex_transform_array = np.conj(np.sqrt((4.0 * np.pi) / ((2.0 * component_degree_vector.astype(np.float64)) + 1.0)) * complex_sh_value_array)
	degree_reversed_complex_transform_vector = complex_transform_array[..., component_index_vector - (2 * component_order_vector)]

	# Convert to real transform vector.
	order_positive_mask = (component_order_vector > 0)
	order_negative_mask = (component_order_vector < 0)
	order_zero_mask = (component_order_vector == 0)
	condon_shortley_vector = np.power(-1.0, component_order_vector.astype(np.float64))

	real_transform_array = order_zero_mask * np.real(complex_transform_array) + (1.0 / np.sqrt(2.0)) * (
		(order_positive_mask * np.real(complex_transform_array)) + 
		(order_positive_mask * condon_shortley_vector * np.real(degree_reversed_complex_transform_vector)) + 
		(order_negative_mask * np.imag(degree_reversed_complex_transform_vector)) + 
		(order_negative_mask * condon_shortley_vector * (-1.0) * np.imag(complex_transform_array))
	)

	return real_transform_array


class SHMixture(object):
	"""
	Class representing a mixture of spherical harmonics
	"""
	def __init__(self):
		self._all_weight_vector = np.zeros((1, ), dtype=np.float64)


	def _is_capacity_enough_for_degree(self, degree):
		return (self._all_weight_vector.size >= ((degree + 1) * (degree + 1)))
	

	def _expand_capacity_to_degree(self, degree):
		old_capacity = self._all_weight_vector.size
		assert (old_capacity < ((degree + 1) * (degree + 1)))

		new_capacity = ((degree + 1) * (degree + 1))
		new_all_weight_vector = np.zeros((new_capacity, ), dtype=np.float64)
		new_all_weight_vector[0:old_capacity] = self._all_weight_vector[:]
		self._all_weight_vector = new_all_weight_vector
	

	def copy(self):
		copied_instance = SHMixture()
		copied_instance.set_all_weights(self._all_weight_vector)

		return copied_instance


	def get_all_weights(self):
		return np.copy(self._all_weight_vector)
	

	def set_all_weights(self, all_weight_vector):
		all_weight_vector = np.array(all_weight_vector, dtype=np.float64)
		assert (all_weight_vector.ndim == 1)
		
		self._all_weight_vector = all_weight_vector
	

	def get_degree_weights(self, degree):
		degree = int(degree)
		assert (degree >= 0)
		(degree_weight_index_begin, degree_weight_index_end) = sh_degree_to_linear_index_range(degree)
		assert (self._all_weight_vector.size >= degree_weight_index_end)

		return np.copy(self._all_weight_vector[degree_weight_index_begin:degree_weight_index_end])
	

	def set_degree_weights(self, degree, degree_weight_vector):
		degree = int(degree)
		assert (degree >= 0)
		degree_weight_vector = np.array(degree_weight_vector, dtype=np.float64)
		assert (degree_weight_vector.ndim == 1)
		assert (degree_weight_vector.size == (2 * degree + 1))

		if (not self._is_capacity_enough_for_degree(degree)):
			self._expand_capacity_to_degree(degree)
		
		(degree_weight_index_begin, degree_weight_index_end) = sh_degree_to_linear_index_range(degree)
		self._all_weight_vector[degree_weight_index_begin:degree_weight_index_end] = degree_weight_vector[:]
	

	def get_single_weight(self, degree, order):
		degree = int(degree)
		assert (degree >= 0)
		order = int(order)
		assert (abs(order) <= degree)
		weight_index = sh_degree_and_order_to_linear_index(degree, order)
		assert (self._all_weight_vector.size >= (weight_index + 1))

		return self._all_weight_vector[weight_index]


	def set_single_weight(self, degree, order, weight):
		degree = int(degree)
		assert (degree >= 0)
		order = int(order)
		assert (abs(order) <= degree)
		weight = float(weight)

		if (not self._is_capacity_enough_for_degree(degree)):
			self._expand_capacity_to_degree(degree)
		
		weight_index = sh_degree_and_order_to_linear_index(degree, order)
		self._all_weight_vector[weight_index] = weight
	

	def evaluate(self, azimuth_array, colatitude_array):
		if (not isinstance(azimuth_array, np.ndarray)):
			azimuth_array = np.array(azimuth_array, dtype=np.float64)
		assert np.issubdtype(azimuth_array.dtype, np.floating)
		num_input_dimensions = azimuth_array.ndim

		if (not isinstance(colatitude_array, np.ndarray)):
			colatitude_array = np.array(colatitude_array, dtype=np.float64)
		assert np.issubdtype(colatitude_array.dtype, np.floating)
		assert (colatitude_array.ndim == num_input_dimensions)

		weight_index_vector = np.arange(0, self._all_weight_vector.size, 1, dtype=np.int32)
		(weight_degree_vector, weight_order_vector) = sh_linear_index_to_degree_and_order(weight_index_vector)
		
		# Based on https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html, broadcasting starts from trailing dimenisions.
		# This means if the dimensions azimuth_array and colatitude_array are expanded by 1 at the end, any broadcasting-compatible
		# operation will broadcast weight_degree_vector and weight_order_vector across the newly expanded dimension.
		component_value_array = sh_evaluate(
			weight_degree_vector, 
			weight_order_vector, 
			np.expand_dims(azimuth_array, num_input_dimensions), 
			np.expand_dims(colatitude_array, num_input_dimensions)
		)

		mixture_value_array = np.sum(
			self._all_weight_vector * component_value_array, axis=num_input_dimensions, dtype=np.float64, keepdims=False
		)
		
		return mixture_value_array
	

	def apply_rotation(self, rotation_matrix):
		assert util.transform.is_rotation_matrix(rotation_matrix)

		num_weights = self._all_weight_vector.size
		(max_degree, _) = sh_linear_index_to_degree_and_order(num_weights - 1)
		transform_matrix = sh_rotation_transform(rotation_matrix, max_degree)[:, 0:num_weights]

		rotated_weight_vector = np.reshape(transform_matrix.dot(np.reshape(self._all_weight_vector, (num_weights, 1))), (num_weights, ))
		self._all_weight_vector = rotated_weight_vector
	

	def detect_z_rotation_symmetry(self, absolute_tolerance, relative_tolerance):
		"""
		Detect discrete or continuous z-rotation symmetry around the z-axis.

		All zonal spherical harmonic mixtures (i.e. all components have order 0) exhibit continuous z-rotation symmetry.

		Given a mixture of spherical harmonics, it must exhibit 1-fold discrete discrete z-rotation symmetry (i.e. rotation by 2\pi), but 
		we are not interested in that. We are interested in what is the maximum-fold (minimum angle) discrete z-rotation symmetry of the mixture.

		Order-M (M != 0) spherical harmonics exhibit M-fold discrete z-rotation symmetry. Let O be the set of orders of all non-zonal, non-zero 
		components in the mixture, then the mixture exhibits greatest_common_divisor(O)-fold discrete z-rotation symmetry.

		Inputs:
			absolute_tolerance: Weights whose absolute values are larger than absolute_tolerance are considered "nonzero".
			relative_tolerance: Weights that satisfy (abs(weight) > (max_abs_weight * relative_tolerance)) are considered "nonzero".
		
		Output:
			is_zonal              : Boolean indicating whether or not this mixture is zonal (i.e. exhibits continuous z-rotation symmetry).
			discrete_symmetry_fold: If the mixture is not zonal, this is the largest fold for the discrete symmetry.
			                        If the mixture is zonal, this is meaningless and the returned value is not specified.

		References:
			Accurate detection of symmetries in 3D shapes, Martinet et al., 2006
		
		Notes:
			The claims in the description of this function can be verified using Equation (6) of the reference paper.
			Notice that if \alpha is an angle of symmetry, so is -\alpha.
			# Then the equation can be simplified to C^m_l = cos(m\alpha)C^m_l, which means (C^m_l = 0) or (m\alpha = 2k\pi).
		"""
		absolute_tolerance = float(absolute_tolerance)
		assert (absolute_tolerance >= 0.0)
		
		relative_tolerance = float(relative_tolerance)
		assert (relative_tolerance >= 0.0)
		assert (relative_tolerance <= 1.0)

		weight_index_vector = np.arange(0, self._all_weight_vector.size, 1, dtype=np.int32)
		(_, weight_order_vector) = sh_linear_index_to_degree_and_order(weight_index_vector)
		weight_non_zonal_mask = (weight_order_vector != 0)

		abs_weight_vector = np.abs(self._all_weight_vector)
		max_abs_weight = np.amax(abs_weight_vector, axis=None, keepdims=False)
		weight_non_zero_mask = np.logical_or(
			(abs_weight_vector > absolute_tolerance), 
			(abs_weight_vector > (max_abs_weight * relative_tolerance))
		)

		# A "contributing order" is defined as an order of a non-zero, non-zonal spherical harmonic component.
		contributing_order_vector = weight_order_vector[np.logical_and(weight_non_zonal_mask, weight_non_zero_mask)]
		if (contributing_order_vector.size == 0):
			# The mixture is zonal. Discrete symmetry is meaningless in this case.
			return (True, None)
		
		discrete_symmetry_fold = abs(contributing_order_vector[0])
		for order_index in range(1, contributing_order_vector.size):
			discrete_symmetry_fold = math.gcd(discrete_symmetry_fold, abs(contributing_order_vector[order_index]))

		return (False, discrete_symmetry_fold)