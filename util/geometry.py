from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np


#################################
# Point
#################################

def compute_points_distance(point_matrix1, point_matrix2, verbose=False):
	"""
	Given two sets of points, compute the pairwise distance between each pair.

	This function is equivalent to scipy.spatial.distance.cdist(point_matrix1, metric="euclidean"),  
	but the vectorized implementation is 10 times faster for large arrays. See geometry_test.py for more details.

	Inputs:
		point_matrix1: A [num_points, num_dimensions] matrix containing the first set of points.
		point_matrix2: A [num_points, num_dimensions] matrix containing the second set of points.
		verbose      : A boolean indicating whether or not to print progress
	"""
	assert isinstance(point_matrix1, np.ndarray)
	assert (point_matrix1.ndim == 2)
	assert (np.issubdtype(point_matrix1.dtype, np.integer) or np.issubdtype(point_matrix1.dtype, np.floating))
	(num_points1, num_dimensions) = point_matrix1.shape

	assert isinstance(point_matrix2, np.ndarray)
	assert (point_matrix2.ndim == 2)
	assert (np.issubdtype(point_matrix2.dtype, np.integer) or np.issubdtype(point_matrix2.dtype, np.floating))
	num_points2 = point_matrix2.shape[0]
	assert (point_matrix2.shape[1] == num_dimensions)

	verbose = bool(verbose)

	if (verbose):
		print("[geometry.compute_points_distance] - computing summed squares ... ")
	
	point_sum_squared_vector1 = np.sum(np.square(point_matrix1), axis=1, dtype=point_matrix1.dtype, keepdims=False)
	point_sum_squared_vector2 = np.sum(np.square(point_matrix2), axis=1, dtype=point_matrix2.dtype, keepdims=False)

	if (verbose):
		print("[geometry.compute_points_distance] - computing X.dot(X^T) ...")
	
	x_dot_xT_matrix = np.dot(point_matrix1, np.transpose(point_matrix2))

	if (verbose):
		print("[geometry.compute_points_distance] - computing distance matrix ...")
	
	distance_squared_matrix = (
		np.reshape(point_sum_squared_vector1, (num_points1, 1)) - 
		(2.0 * x_dot_xT_matrix) + 
		np.reshape(point_sum_squared_vector2, (1, num_points2))
	)

	# Due to limited floating point precision, some of the entries may be slightly negative. Truncate them to 0.
	distance_squared_matrix = np.maximum(distance_squared_matrix, 0.0)
	distance_matrix = np.sqrt(distance_squared_matrix)
	
	return distance_matrix


def compute_points_pairwise_distance(point_matrix, verbose=False):
	"""
	Given some points, compute the pairwise distance between each pair.

	This function is equivalent to scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(point_matrix, metric="euclidean")), 
	but the vectorized implementation is 10 times faster for large arrays. See geometry_test.py for more details.

	Inputs:
		point_matrix: A [num_points, num_dimensions] matrix containing the points.
		verbose     : A boolean indicating whether or not to print progress
	"""
	assert isinstance(point_matrix, np.ndarray)
	assert (point_matrix.ndim == 2)
	assert (np.issubdtype(point_matrix.dtype, np.integer) or np.issubdtype(point_matrix.dtype, np.floating))
	(num_points, num_dimensions) = point_matrix.shape

	verbose = bool(verbose)
	
	distance_matrix = compute_points_distance(point_matrix, point_matrix, verbose)

	# Due to limited floating point precision, the matrix may not be symmetric, and the diagonal may not be 0.
	# Enforce these constraints manually
	if (verbose):
		print("[geometry.compute_points_pairwise_distance] - cleaning pairwise distancce matrix ...")
	
	distance_matrix = (distance_matrix + np.transpose(distance_matrix)) / 2.0
	np.fill_diagonal(distance_matrix, 0.0)
	
	return distance_matrix


#################################
# Triangle
#################################

def compute_triangles_area(triangle_vertex_tensor):
	"""
	Given an array of triangles in N-dimensional space, compute their areas.

	Inputs:
		triangle_vertex_tensor: A [num_triangles, 3, num_dimensions] tensor containing the triangle vertex coordinates.
	
	Outputs:
		triangle_area_vector: A [num_triangles] vector containing the areas of triangles.
	
	Reference:
		https://en.wikipedia.org/wiki/Triangle#Using_vectors
	"""
	assert isinstance(triangle_vertex_tensor, np.ndarray)
	assert (triangle_vertex_tensor.ndim == 3)
	assert (triangle_vertex_tensor.shape[1] == 3)
	assert (triangle_vertex_tensor.shape[2] >= 2)

	# Using notation in the reference
	triangle_AB_matrix = triangle_vertex_tensor[:, 1, :] - triangle_vertex_tensor[:, 0, :]
	triangle_AC_matrix = triangle_vertex_tensor[:, 2, :] - triangle_vertex_tensor[:, 0, :]

	triangle_area_vector = 0.5 * np.sqrt(
		np.sum(np.square(triangle_AB_matrix), axis=1, dtype=np.float64, keepdims=False) * 
		np.sum(np.square(triangle_AC_matrix), axis=1, dtype=np.float64, keepdims=False) - 
		np.square(np.sum(triangle_AB_matrix * triangle_AC_matrix, axis=1, dtype=np.float64, keepdims=False))
	)

	return triangle_area_vector


#################################
# Rectangle
#################################

class RectangleShape(object):
	def __init__(self, size_x, size_y):
		self.size_x = size_x
		self.size_y = size_y
	

	def __repr__(self):
		return "RectangleShape(size_x=%r, size_y=%r)" % (self.size_x, self.size_y)
	

	@property
	def size_x(self):
		return self._size_x
	
	@size_x.setter
	def size_x(self, value):
		value = float(value)
		assert (value >= 0.0)
		self._size_x = value
	

	@property
	def size_y(self):
		return self._size_y
	
	@size_y.setter
	def size_y(self, value):
		value = float(value)
		assert (value >= 0.0)
		self._size_y = value
	

	def copy(self):
		return RectangleShape(self.size_x, self.size_y)
	

	def apply_scale(self, scale_factor):
		scale_factor_float = float(scale_factor)
		assert (scale_factor_float >= 0.0)

		new_size_x = self._size_x * scale_factor_float
		new_size_y = self._size_y * scale_factor_float

		self.size_x = new_size_x
		self.size_y = new_size_y


class RectangleXYWHParam(object):
	"""
	References:
		1. https://arxiv.org/abs/1504.08083
	"""
	def __init__(self, param_tx, param_ty, param_tw, param_th):
		self.param_tx = param_tx
		self.param_ty = param_ty
		self.param_tw = param_tw
		self.param_th = param_th
	

	def __repr__(self):
		return "RectangleXYWHParam(param_tx=%r, param_ty=%r, param_tw=%r, param_th=%r)" % (
			self.param_tx, self.param_ty, self.param_tw, self.param_th
		)
	

	@property
	def param_tx(self):
		return self._param_tx
	
	@param_tx.setter
	def param_tx(self, value):
		value = float(value)
		self._param_tx = value
	

	@property
	def param_ty(self):
		return self._param_ty
	
	@param_ty.setter
	def param_ty(self, value):
		value = float(value)
		self._param_ty = value
	

	@property
	def param_tw(self):
		return self._param_tw
	
	@param_tw.setter
	def param_tw(self, value):
		value = float(value)
		self._param_tw = value
	

	@property
	def param_th(self):
		return self._param_th
	
	@param_th.setter
	def param_th(self, value):
		value = float(value)
		self._param_th = value
	

	@property
	def vector_form(self):
		return np.array([self.param_tx, self.param_ty, self.param_tw, self.param_th], dtype=np.float32)
	

	def copy(self):
		return RectangleXYWHParam(self.param_tx, self.param_ty, self.param_tw, self.param_th)
	

	def get_post_composition(self, other_param):
		assert isinstance(other_param, RectangleXYWHParam)

		composed_param_tx = self.param_tx + (np.exp(self.param_tw) * other_param.param_tx)
		composed_param_ty = self.param_ty + (np.exp(self.param_th) * other_param.param_ty)
		composed_param_tw = self.param_tw + other_param.param_tw
		composed_param_th = self.param_th + other_param.param_th

		return RectangleXYWHParam(composed_param_tx, composed_param_ty, composed_param_tw, composed_param_th)
	

	def get_inverse(self):
		inverted_param_tx = (-1.0 * self.param_tx / np.exp(self.param_tw))
		inverted_param_ty = (-1.0 * self.param_ty / np.exp(self.param_th))
		inverted_param_tw = (-1.0 * self.param_tw)
		inverted_param_th = (-1.0 * self.param_th)

		return RectangleXYWHParam(inverted_param_tx, inverted_param_ty, inverted_param_tw, inverted_param_th)


class RectangleShiftParam(object):
	def __init__(self, shift_top, shift_left, shift_bottom, shift_right):
		self.set_shift_top_and_bottom(shift_top, shift_bottom)
		self.set_shift_left_and_right(shift_left, shift_right)
	

	@property
	def shift_top(self):
		return self._shift_top
	
	@shift_top.setter
	def shift_top(self, value):
		value = float(value)
		assert ((value + self.shift_bottom) >= -1.0)
		self._shift_top = value
	

	@property
	def shift_left(self):
		return self._shift_left
	
	@shift_left.setter
	def shift_left(self, value):
		value = float(value)
		assert ((value + self.shift_right) >= -1.0)
		self._shift_left = value
	

	@property
	def shift_bottom(self):
		return self._shift_bottom
	
	@shift_bottom.setter
	def shift_bottom(self, value):
		value = float(value)
		assert ((value + self.shift_top) >= -1.0)
		self._shift_bottom = value
	

	@property
	def shift_right(self):
		return self._shift_right
	
	@shift_right.setter
	def shift_right(self, value):
		value = float(value)
		assert ((value + self.shift_left) >= -1.0)
		self._shift_right = value
	

	def set_shift_top_and_bottom(self, top_value, bottom_value):
		top_value = float(top_value)
		bottom_value = float(bottom_value)
		assert ((top_value + bottom_value) >= -1.0)
		self._shift_top = top_value
		self._shift_bottom = bottom_value
	

	def set_shift_left_and_right(self, left_value, right_value):
		left_value = float(left_value)
		right_value = float(right_value)
		assert ((left_value + right_value) >= -1.0)
		self._shift_left = left_value
		self._shift_right = right_value
	

	def get_post_composition(self, other_param):
		assert isinstance(other_param, RectangleShiftParam)
		shifted_relative_width = 1.0 + self.shift_left + self.shift_right
		shifted_relative_height = 1.0 + self.shift_top + self.shift_bottom

		composed_shift_top = self.shift_top + (shifted_relative_height * other_param.shift_top)
		composed_shift_left = self.shift_left + (shifted_relative_width * other_param.shift_left)
		composed_shift_bottom = self.shift_bottom + (shifted_relative_height * other_param.shift_bottom)
		composed_shift_right = self.shift_right + (shifted_relative_width * other_param.shift_right)

		return RectangleShiftParam(composed_shift_top, composed_shift_left, composed_shift_bottom, composed_shift_right)
	

	def get_inverse(self):
		shifted_relative_width = 1.0 + self.shift_left + self.shift_right
		shifted_relative_height = 1.0 + self.shift_top + self.shift_bottom

		inverted_shift_top = -self.shift_top / shifted_relative_height
		inverted_shift_left = -self.shift_left / shifted_relative_width
		inverted_shift_bottom = -self.shift_bottom / shifted_relative_height
		inverted_shift_right = -self.shift_right / shifted_relative_width

		return RectangleShiftParam(inverted_shift_top, inverted_shift_left, inverted_shift_bottom, inverted_shift_right)
	

	@property
	def vector_form(self):
		return np.array([self.shift_top, self.shift_left, self.shift_bottom, self.shift_right], dtype=np.float32)
	

	def copy(self):
		return RectangleShiftParam(self.shift_top, self.shift_left, self.shift_bottom, self.shift_right)


class Rectangle(object):
	def __init__(self, min_x, min_y, size_x, size_y):
		self.min_x = min_x
		self.min_y = min_y
		self.shape = RectangleShape(size_x, size_y)
	

	def __repr__(self):
		return "Rectangle(min_x=%r, min_y=%r, size_x=%r, size_y=%r)" %(
			self.min_x, self.min_y, self.size_x, self.size_y
		)
	

	@property
	def min_x(self):
		return self._min_x
	
	@min_x.setter
	def min_x(self, value):
		self._min_x = float(value)
	

	@property
	def min_y(self):
		return self._min_y
	
	@min_y.setter
	def min_y(self, value):
		self._min_y = float(value)
	

	@property
	def shape(self):
		return self._shape
	
	@shape.setter
	def shape(self, value):
		assert isinstance(value, RectangleShape)
		self._shape = value
	

	@property
	def size_x(self):
		return self._shape.size_x
	
	@size_x.setter
	def size_x(self, value):
		self._shape.size_x = value
	

	@property
	def size_y(self):
		return self._shape.size_y
	
	@size_y.setter
	def size_y(self, value):
		self._shape.size_y = value
	

	def copy(self):
		return Rectangle(self.min_x, self.min_y, self.size_x, self.size_y)
	

	@property
	def max_size(self):
		if (self.size_x >= self.size_y):
			return self.size_x
		else:
			return self.size_y
	

	@property
	def min_size(self):
		if (self.size_x <= self.size_y):
			return self.size_x
		else:
			return self.size_y
	

	@property
	def max_x(self):
		return (self.min_x + self.size_x)
	

	@property
	def max_y(self):
		return (self.min_y + self.size_y)
	

	@property
	def center_x(self):
		return (self.min_x + (self.size_x / 2.0))
	

	@property
	def center_y(self):
		return (self.min_y + (self.size_y / 2.0))
	

	def contains_rectangle(self, other_rectangle):
		assert isinstance(other_rectangle, Rectangle)
		return (
			(other_rectangle.min_x >= self.min_x) and (other_rectangle.max_x <= self.max_x) and
			(other_rectangle.min_y >= self.min_y) and (other_rectangle.max_y <= self.max_y)
		)
	

	def strictly_contains_rectangle(self, other_rectangle):
		assert isinstance(other_rectangle, Rectangle)
		return (
			(other_rectangle.min_x > self.min_x) and (other_rectangle.max_x < self.max_x) and
			(other_rectangle.min_y > self.min_y) and (other_rectangle.max_y < self.max_y)
		)
	

	def compute_xywh_param_to_rectangle(self, other_rectangle):
		assert isinstance(other_rectangle, Rectangle)
		assert (self.size_x > 0.0)
		assert (self.size_y > 0.0)
		assert (other_rectangle.size_x > 0.0)
		assert (other_rectangle.size_y > 0.0)

		param_tx = ((other_rectangle.center_x - self.center_x) / self.size_x)
		param_ty = ((other_rectangle.center_y - self.center_y) / self.size_y)
		param_tw = np.log(other_rectangle.size_x / self.size_x)
		param_th = np.log(other_rectangle.size_y / self.size_y)

		return RectangleXYWHParam(param_tx, param_ty, param_tw, param_th)
	

	def compute_shift_param_to_rectangle(self, other_rectangle):
		assert isinstance(other_rectangle, Rectangle)
		assert (self.size_x > 0.0)
		assert (self.size_y > 0.0)

		shift_top = (self.min_y - other_rectangle.min_y) / self.size_y
		shift_left = (self.min_x - other_rectangle.min_x) / self.size_x
		shift_bottom = (other_rectangle.max_y - self.max_y) / self.size_y
		shift_right = (other_rectangle.max_x - self.max_x) / self.size_x

		return RectangleShiftParam(shift_top, shift_left, shift_bottom, shift_right)


	def apply_translation(self, translation_x, translation_y):
		translation_x = float(translation_x)
		translation_y = float(translation_y)

		self.min_x = self.min_x + translation_x
		self.min_y = self.min_y + translation_y
	

	def apply_scale_fixing_min_corner(self, scale_factor):
		scale_factor_float = float(scale_factor)
		assert (scale_factor_float >= 0.0)

		self._shape.apply_scale(scale_factor_float)
	

	def apply_scale_fixing_centroid(self, scale_factor):
		scale_factor_float = float(scale_factor)
		assert (scale_factor_float >= 0.0)

		self.min_x = self.min_x + (self.size_x * (1.0 - scale_factor) / 2.0)
		self.min_y = self.min_y + (self.size_y * (1.0 - scale_factor) / 2.0)
		self._shape.apply_scale(scale_factor_float)
	

	def apply_xywh_param(self, xywh_param):
		assert isinstance(xywh_param, RectangleXYWHParam)

		new_center_x = self.center_x + (xywh_param.param_tx * self.size_x)
		new_center_y = self.center_y + (xywh_param.param_ty * self.size_y)
		new_size_x = self.size_x * np.exp(xywh_param.param_tw)
		new_size_y = self.size_y * np.exp(xywh_param.param_th)

		self.min_x = new_center_x - (new_size_x / 2.0)
		self.min_y = new_center_y - (new_size_y / 2.0)
		self.size_x = new_size_x
		self.size_y = new_size_y
	

	def apply_shift_param(self, shift_param):
		assert isinstance(shift_param, RectangleShiftParam)

		old_size_x = self.size_x
		old_size_y = self.size_y

		self.min_y -= (shift_param.shift_top * old_size_y)
		self.min_x -= (shift_param.shift_left * old_size_x)
		self.size_y += ((shift_param.shift_top + shift_param.shift_bottom) * old_size_y)
		self.size_x += ((shift_param.shift_left + shift_param.shift_right) * old_size_x)
	

	def apply_shift_ratios(self, shift_top, shift_left, shift_bottom, shift_right):
		# RectangleShiftParam construct does all argument checking.
		# This is is why I don't do the math here and have apply_shift_param() call this function.
		shift_param = RectangleShiftParam(shift_top, shift_left, shift_bottom, shift_right)
		self.apply_shift_param(shift_param)
	

	def expand_to_square(self):
		xy_size_difference = self.size_x - self.size_y
		if(xy_size_difference >= 0.0):
			square_size = self.size_x
			square_min_x = self._min_x
			square_min_y = self.min_y - abs(xy_size_difference / 2.0)
		else:
			square_size = self.size_y
			square_min_x = self._min_x - abs(xy_size_difference / 2.0)
			square_min_y = self.min_y
		
		self.min_x = square_min_x
		self.min_y = square_min_y
		self.size_x = square_size
		self.size_y = square_size


def boundaries_to_rectangle(min_x, min_y, max_x, max_y):
	min_x = float(min_x)
	min_y = float(min_y)
	max_x = float(max_x)
	assert (max_x >= min_x)
	max_y = float(max_y)
	assert (max_y >= min_y)

	return Rectangle(min_x, min_y, max_x - min_x, max_y - min_y)


def points_to_bounding_rectangle(x_array, y_array):
	assert isinstance(x_array, np.ndarray)
	assert (x_array.ndim == 1)
	assert (x_array.size >= 1)
	assert isinstance(y_array, np.ndarray)
	assert (y_array.ndim == 1)
	assert (y_array.size == x_array.size)

	min_x = np.amin(x_array, axis=None)
	min_y = np.amin(y_array, axis=None)
	max_x = np.amax(x_array, axis=None)
	max_y = np.amax(y_array, axis=None)
	bounding_rectangle = boundaries_to_rectangle(min_x, min_y, max_x, max_y)

	return bounding_rectangle


def nonzero_bounding_rectangle(input_matrix):
	assert isinstance(input_matrix, np.ndarray)
	assert (input_matrix.ndim == 2)

	(nonzero_y_array, nonzero_x_array) = np.nonzero(input_matrix)
	nonzero_bounding_rectangle = points_to_bounding_rectangle(nonzero_x_array, nonzero_y_array)

	return nonzero_bounding_rectangle


#################################
# Sphere
#################################

def compute_points_azimuth(point_array):
	if (not isinstance(point_array, np.ndarray)):
		point_array = np.array(point_array, dtype=np.float64)
	assert np.issubdtype(point_array.dtype, np.floating)
	assert (point_array.ndim >= 1)
	assert (point_array.shape[-1] == 3)

	azimuth_array = np.arctan2(point_array[..., 1], point_array[..., 0])

	return azimuth_array


def compute_points_colatitude(point_array, division_epsilon=1e-12):
	if (not isinstance(point_array, np.ndarray)):
		point_array = np.array(point_array, dtype=np.float64)
	assert np.issubdtype(point_array.dtype, np.floating)
	assert (point_array.ndim >= 1)
	assert (point_array.shape[-1] == 3)

	division_epsilon = float(division_epsilon)
	assert (division_epsilon > 0)

	norm_array = np.linalg.norm(point_array, ord=2, axis=-1, keepdims=False)
	colatitude_array = np.arccos(point_array[..., 2] / (norm_array + division_epsilon))

	return colatitude_array


def compute_points_elevation(point_array, division_epsilon=1e-12):
	if (not isinstance(point_array, np.ndarray)):
		point_array = np.array(point_array, dtype=np.float64)
	assert np.issubdtype(point_array.dtype, np.floating)
	assert (point_array.ndim >= 1)
	assert (point_array.shape[-1] == 3)

	division_epsilon = float(division_epsilon)
	assert (division_epsilon > 0)

	norm_array = np.linalg.norm(point_array, ord=2, axis=-1, keepdims=False)
	elevation_array = np.arcsin(point_array[..., 2] / (norm_array + division_epsilon))

	return elevation_array


#################################
# Other
#################################


def get_dodecahedron_vertices():
	"""
	Get the normalized verticex coordinates of a regular dodecahedron.

	Outputs
		vertex_matrix: A [20, 3] matrix containing the normalized vertex coordinates of a regular dodecahedron.
	
	References:
		https://en.wikipedia.org/wiki/Regular_dodecahedron#Cartesian_coordinates
	"""

	golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
	
	vertex_matrix = np.array([
		[-1.0, -1.0, -1.0], 
		[-1.0, -1.0,  1.0], 
		[-1.0,  1.0, -1.0], 
		[-1.0,  1.0,  1.0], 
		[ 1.0, -1.0, -1.0], 
		[ 1.0, -1.0,  1.0], 
		[ 1.0,  1.0, -1.0], 
		[ 1.0,  1.0,  1.0], 
		[0.0, -golden_ratio, -1.0 / golden_ratio], 
		[0.0, -golden_ratio,  1.0 / golden_ratio], 
		[0.0,  golden_ratio, -1.0 / golden_ratio], 
		[0.0,  golden_ratio,  1.0 / golden_ratio], 
		[-1.0 / golden_ratio, 0.0, -golden_ratio], 
		[-1.0 / golden_ratio, 0.0,  golden_ratio], 
		[ 1.0 / golden_ratio, 0.0, -golden_ratio], 
		[ 1.0 / golden_ratio, 0.0,  golden_ratio], 
		[-golden_ratio, -1.0 / golden_ratio, 0.0], 
		[-golden_ratio,  1.0 / golden_ratio, 0.0], 
		[ golden_ratio, -1.0 / golden_ratio, 0.0], 
		[ golden_ratio,  1.0 / golden_ratio, 0.0], 
	], dtype=np.float64)

	normalized_vertex_matrix = vertex_matrix / np.linalg.norm(vertex_matrix, ord=2, axis=1, keepdims=True)
	
	return normalized_vertex_matrix


def get_icosahedron_vertices():
	"""
	Get the normalized vertex coordinates of a regular icosahedron.

	Outputs
		vertex_matrix: A [12, 3] matrix containing thenormalized vertex coordinates of a regular icosahedron.
	
	References:
		https://en.wikipedia.org/wiki/Regular_dodecahedron#Cartesian_coordinates
	"""

	golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
	
	vertex_matrix = np.array([
		[0.0,  1.0,  golden_ratio], 
		[0.0,  1.0, -golden_ratio], 
		[0.0, -1.0,  golden_ratio], 
		[0.0, -1.0, -golden_ratio], 
		[ 1.0,  golden_ratio, 0.0], 
		[ 1.0, -golden_ratio, 0.0], 
		[-1.0,  golden_ratio, 0.0], 
		[-1.0, -golden_ratio, 0.0], 
		[ golden_ratio, 0.0,  1.0], 
		[ golden_ratio, 0.0, -1.0], 
		[-golden_ratio, 0.0,  1.0], 
		[-golden_ratio, 0.0, -1.0], 
	], dtype=np.float64)

	normalized_vertex_matrix = vertex_matrix / np.linalg.norm(vertex_matrix, ord=2, axis=1, keepdims=True)

	return normalized_vertex_matrix