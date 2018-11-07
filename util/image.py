from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import skimage.color
import skimage.feature
import skimage.transform

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, ".."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import util.geometry


def image_to_rgb_float(image_array, background_pixel=(1.0, 1.0, 1.0)):
	assert isinstance(image_array, np.ndarray)
	assert (image_array.ndim >= 2)
	assert (image_array.ndim <= 3)
	if ((image_array.ndim == 3) and image_array.shape[2] == 1):
		image_array = np.squeeze(image_array, axis=2)

	if (image_array.ndim == 2):
		rgb_image_array = skimage.color.gray2rgb(image_array)
	else:
		if (image_array.shape[2] == 3):
			rgb_image_array = image_array
		elif (image_array.shape[2] == 4):
			rgb_image_array = skimage.color.rgba2rgb(image_array, background=background_pixel)
		else:
			raise Exception("The input 'image_array' has invalid shape %s" % (image_array.shape, ))

	rgb_image_float_array = skimage.img_as_float(rgb_image_array)
	return rgb_image_float_array


def image_to_rgba_float(image_array):
	assert isinstance(image_array, np.ndarray)
	assert (image_array.ndim >= 2)
	assert (image_array.ndim <= 3)
	image_size_y = image_array.shape[0]
	image_size_x = image_array.shape[1]
	if ((image_array.ndim == 3) and image_array.shape[2] == 1):
		image_array = np.squeeze(image_array, axis=2)
	image_float_array = skimage.img_as_float(image_array)

	if (image_float_array.ndim == 2):
		rgba_image_float_array = np.concatenate(
			[skimage.color.gray2rgb(image_float_array), np.ones((image_size_y, image_size_x, 1), dtype=image_float_array.dtype)], axis=2
		)
	else:
		if (image_float_array.shape[2] == 3):
			rgba_image_float_array = np.concatenate(
				[image_float_array, np.ones((image_size_y, image_size_x, 1), dtype=image_float_array.dtype)], axis=2
			)
		elif (image_float_array.shape[2] == 4):
			rgba_image_float_array = image_float_array
		else:
			raise Exception("The input 'image_array' has invalid shape %s" % (image_array.shape, ))
	
	return rgba_image_float_array


def rgba_to_gray(rgba_image_array, background_intensity):
	assert isinstance(rgba_image_array, np.ndarray)
	assert (rgba_image_array.ndim == 3)
	assert (rgba_image_array.shape[2] == 4)
	image_float_array = skimage.img_as_float(rgba_image_array)

	background_intensity = float(background_intensity)
	assert (background_intensity >= 0.0)
	assert (background_intensity <= 1.0)

	alpha_array = image_float_array[:, :, 3]
	gray_image_float_array = (
		(skimage.color.rgb2gray(image_float_array[:, :, 0:3]) * alpha_array) + 
		(background_intensity * (1.0 - alpha_array))
	)

	return gray_image_float_array


def blend_rgba(front_rgba_image_array, back_rgba_image_array):
	assert isinstance(front_rgba_image_array, np.ndarray)
	assert (front_rgba_image_array.ndim == 3)
	assert (front_rgba_image_array.shape[2] == 4)
	front_rgba_image_float_array = skimage.img_as_float(front_rgba_image_array)

	assert isinstance(back_rgba_image_array, np.ndarray)
	assert (back_rgba_image_array.shape == front_rgba_image_array.shape)
	back_rgba_image_float_array = skimage.img_as_float(back_rgba_image_array)

	# Clip background alpha by a very small value, so when both alpha values are 0.0, 
	# the algortihm gives the back color, instead of inf or nan.
	front_alpha_array = front_rgba_image_float_array[:, :, 3, np.newaxis]
	back_alpha_array = np.maximum(1e-5, back_rgba_image_float_array[:, :, 3, np.newaxis])

	mixed_alpha_array = front_alpha_array + ((1.0 - front_alpha_array) * back_alpha_array)
	mixed_color_array = ((
		(front_rgba_image_float_array[:, :, 0:3] * front_alpha_array) + 
		(back_rgba_image_float_array[:, :, 0:3] * (1.0 - front_alpha_array) * back_alpha_array)
	) / mixed_alpha_array)

	mixed_rgba_image_float_array = np.concatenate([mixed_color_array, mixed_alpha_array], axis=2)

	return mixed_rgba_image_float_array


def rectangle_to_image_index(rectangle):
	assert isinstance(rectangle, util.geometry.Rectangle)
	assert (rectangle.min_x >= 0.0)
	assert (rectangle.min_y >= 0.0)

	# Two advantages of this conversion:
	# 	1. As long as rectangle is within boundary, the converted indices are also within boundary.
	# 	2. Rectangles with same float width will lead to image regions with same integer width.
	image_min_x = int(np.floor(rectangle.min_x))
	image_max_x = (image_min_x + int(np.around(rectangle.size_x)))
	image_min_y = int(np.floor(rectangle.min_y))
	image_max_y = (image_min_y + int(np.around(rectangle.size_y)))

	# array[index_object] is equivalent to array[image_min_y:image_max_y, image_min_x:image_max_x, ...]
	# The Ellipsis in the end allows this to work on both [H, W] images and [H, W, C] images.
	index_object = (slice(image_min_y, image_max_y, 1), slice(image_min_x, image_max_x, 1), Ellipsis)

	return index_object


def randomly_crop_to_size(image_array, crop_size_x, crop_size_y):
	assert isinstance(image_array, np.ndarray)
	assert (image_array.ndim >= 2)
	assert (image_array.ndim <= 3)
	image_size_y = image_array.shape[0]
	image_size_x = image_array.shape[1]

	crop_size_x = int(crop_size_x)
	assert (crop_size_x >= 0)
	assert (crop_size_x <= image_size_x)

	crop_size_y = int(crop_size_y)
	assert (crop_size_y >= 0)
	assert (crop_size_y <= image_size_y)

	crop_min_x = np.random.randint(0, (image_size_x - crop_size_x + 1))
	crop_min_y = np.random.randint(0, (image_size_y - crop_size_y + 1))

	cropped_image_array = image_array[crop_min_y:(crop_min_y + crop_size_y), crop_min_x:(crop_min_x + crop_size_x), ...]

	return (cropped_image_array, crop_min_x, crop_min_y)


def cut_rgba_transparent_border(rgba_image_array):
	assert isinstance(rgba_image_array, np.ndarray)
	assert (rgba_image_array.ndim == 3)
	assert (rgba_image_array.shape[2] == 4)

	non_transparent_bounding_rectangle = util.geometry.nonzero_bounding_rectangle(rgba_image_array[:, :, 3])
	crop_index_object = rectangle_to_image_index(non_transparent_bounding_rectangle)
	cropped_image_array = rgba_image_array[crop_index_object]

	return cropped_image_array


def scale_long_edge_to_length(image_array, expected_long_size):
	assert isinstance(image_array, np.ndarray)
	assert (image_array.ndim >= 2)
	image_size_y = image_array.shape[0]
	image_size_x = image_array.shape[1]

	expected_long_size = int(expected_long_size)
	assert (expected_long_size > 0)

	if (image_size_x > image_size_y):
		scaled_size_x = expected_long_size
		scaled_size_y = int(np.around(float(image_size_y * expected_long_size) / float(image_size_x)))
	else:
		scaled_size_x = int(np.around(float(image_size_x * expected_long_size) / float(image_size_y)))
		scaled_size_y = expected_long_size
	
	scaled_image_array = skimage.transform.resize(
			image_array, (scaled_size_y, scaled_size_x), order=1,
			mode="constant", preserve_range=False, anti_aliasing=True
	)

	return scaled_image_array


def scale_short_edge_to_length(image_array, expected_short_size):
	assert isinstance(image_array, np.ndarray)
	assert (image_array.ndim >= 2)
	image_size_y = image_array.shape[0]
	image_size_x = image_array.shape[1]

	expected_short_size = int(expected_short_size)
	assert (expected_short_size > 0)

	if (image_size_y > image_size_x):
		scaled_size_x = expected_short_size
		scaled_size_y = int(np.around(float(image_size_y * expected_short_size) / float(image_size_x)))
	else:
		scaled_size_x = int(np.around(float(image_size_x * expected_short_size) / float(image_size_y)))
		scaled_size_y = expected_short_size
	
	scaled_image_array = skimage.transform.resize(
			image_array, (scaled_size_y, scaled_size_x), order=1,
			mode="constant", preserve_range=False, anti_aliasing=True
	)

	return scaled_image_array


class RandomOcclusionConfig(object):
	def __init__(self):
		self._expansion_ratio = 0.2
		self._occlusion_prob = 0.75
		self._occlusion_blob_prob = 0.5
		self._occlusion_blob_middle_prob = 0.2
		self._occlusion_blob_middle_verticle_prob = 0.5
		self._occlusion_blob_middle_min_ratio = 0.1
		self._occlusion_blob_middle_max_ratio = 0.3
		self._occlusion_blob_side_lr_prob = 0.5
		self._occlusion_blob_side_min_ratio = 0.1
		self._occlusion_blob_side_max_ratio = 0.6
		self._occlusion_object_min_ratio = 0.1
		self._occlusion_object_max_ratio = 0.6
	

	@property
	def expansion_ratio(self):
		return self._expansion_ratio
	
	@expansion_ratio.setter
	def expansion_ratio(self, value):
		value = float(value)
		assert (value >= 0.0)
		self._expansion_ratio = value
	

	@property
	def occlusion_prob(self):
		return self._occlusion_prob
	
	@occlusion_prob.setter
	def occlusion_prob(self, value):
		value = float(value)
		assert (value >= 0.0)
		assert (value <= 1.0)
		self._occlusion_prob = value
	

	@property
	def occlusion_blob_prob(self):
		return self._occlusion_blob_prob
	
	@occlusion_blob_prob.setter
	def occlusion_blob_prob(self, value):
		value = float(value)
		assert (value >= 0.0)
		assert (value <= 1.0)
		self._occlusion_blob_prob = value
	

	@property
	def occlusion_blob_middle_prob(self):
		return self._occlusion_blob_middle_prob
	
	@occlusion_blob_middle_prob.setter
	def occlusion_blob_middle_prob(self, value):
		value = float(value)
		assert (value >= 0.0)
		assert (value <= 1.0)
		self._occlusion_blob_middle_prob = value
	

	@property
	def occlusion_blob_middle_verticle_prob(self):
		return self._occlusion_blob_middle_verticle_prob
	
	@occlusion_blob_middle_verticle_prob.setter
	def occlusion_blob_middle_verticle_prob(self, value):
		value = float(value)
		assert (value >= 0.0)
		assert (value <= 1.0)
		self._occlusion_blob_middle_verticle_prob = value
	

	@property
	def occlusion_blob_middle_min_ratio(self):
		return self._occlusion_blob_middle_min_ratio
	
	@occlusion_blob_middle_min_ratio.setter
	def occlusion_blob_middle_min_ratio(self, value):
		value = float(value)
		assert (value >= 0.0)
		assert (value <= 1.0)
		assert (value <= self.occlusion_blob_middle_max_ratio)
		self._occlusion_blob_middle_min_ratio = value
	

	@property
	def occlusion_blob_middle_max_ratio(self):
		return self._occlusion_blob_middle_max_ratio
	
	@occlusion_blob_middle_max_ratio.setter
	def occlusion_blob_middle_max_ratio(self, value):
		value = float(value)
		assert (value >= 0.0)
		assert (value <= 1.0)
		assert (value >= self.occlusion_blob_middle_min_ratio)
		self._occlusion_blob_middle_max_ratio = value
	

	@property
	def occlusion_blob_side_lr_prob(self):
		return self._occlusion_blob_side_lr_prob
	
	@occlusion_blob_side_lr_prob.setter
	def occlusion_blob_side_lr_prob(self, value):
		value = float(value)
		assert (value >= 0.0)
		assert (value <= 1.0)
		self._occlusion_blob_side_lr_prob = value
	

	@property
	def occlusion_blob_side_min_ratio(self):
		return self._occlusion_blob_side_min_ratio
	
	@occlusion_blob_side_min_ratio.setter
	def occlusion_blob_side_min_ratio(self, value):
		value = float(value)
		assert (value >= 0.0)
		assert (value <= 1.0)
		assert (value <= self.occlusion_blob_side_max_ratio)
		self._occlusion_blob_side_min_ratio = value
	

	@property
	def occlusion_blob_side_max_ratio(self):
		return self._occlusion_blob_side_max_ratio
	
	@occlusion_blob_side_max_ratio.setter
	def occlusion_blob_side_max_ratio(self, value):
		value = float(value)
		assert (value >= 0.0)
		assert (value <= 1.0)
		assert (value >= self.occlusion_blob_side_min_ratio)
		self._occlusion_blob_side_max_ratio = value
	

	@property
	def occlusion_object_min_ratio(self):
		return self._occlusion_object_min_ratio
	
	@occlusion_object_min_ratio.setter
	def occlusion_object_min_ratio(self, value):
		value = float(value)
		assert (value >= 0.0)
		assert (value <= 1.0)
		assert (value <= self.occlusion_object_max_ratio)
		self._occlusion_object_min_ratio = value
	

	@property
	def occlusion_object_max_ratio(self):
		return self._occlusion_object_max_ratio
	
	@occlusion_object_max_ratio.setter
	def occlusion_object_max_ratio(self, value):
		value = float(value)
		assert (value >= 0.0)
		assert (value <= 1.0)
		assert (value >= self.occlusion_object_min_ratio)
		self._occlusion_object_max_ratio = value
	

	def copy(self):
		copied_occlusion_config = RandomOcclusionConfig()

		copied_occlusion_config.expansion_ratio = self.expansion_ratio
		copied_occlusion_config.occlusion_prob = self.occlusion_prob
		copied_occlusion_config.occlusion_blob_prob = self.occlusion_blob_prob
		copied_occlusion_config.occlusion_blob_middle_prob = self.occlusion_blob_middle_prob
		copied_occlusion_config.occlusion_blob_middle_verticle_prob = self.occlusion_blob_middle_verticle_prob
		copied_occlusion_config.occlusion_blob_middle_min_ratio = self.occlusion_blob_middle_min_ratio
		copied_occlusion_config.occlusion_blob_middle_max_ratio = self.occlusion_blob_middle_max_ratio
		copied_occlusion_config.occlusion_blob_side_lr_prob = self.occlusion_blob_side_lr_prob
		copied_occlusion_config.occlusion_blob_side_min_ratio = self.occlusion_blob_side_min_ratio
		copied_occlusion_config.occlusion_blob_side_max_ratio = self.occlusion_blob_side_max_ratio
		copied_occlusion_config.occlusion_object_min_ratio = self.occlusion_object_min_ratio
		copied_occlusion_config.occlusion_object_max_ratio = self.occlusion_object_max_ratio

		return copied_occlusion_config


class RandomOcclusionResult(object):
	def __init__(self, size_x, size_y):
		size_x = int(size_x)
		assert (size_x > 0)
		self._size_x = size_x

		size_y = int(size_y)
		assert (size_y > 0)
		self._size_y = size_y
	

	@property
	def size_x(self):
		return self._size_x
	

	@property
	def size_y(self):
		return self._size_y
	

	@property
	def rgba_image_float_array(self):
		return self._rgba_image_float_array
	
	@rgba_image_float_array.setter
	def rgba_image_float_array(self, value):
		assert isinstance(value, np.ndarray)
		assert np.issubdtype(value.dtype, np.floating)
		assert (value.ndim == 3)
		assert (value.shape == (self.size_y, self.size_x, 4))

		self._rgba_image_float_array = value
	

	@property
	def full_mask(self):
		return self._full_mask
	
	@full_mask.setter
	def full_mask(self, value):
		assert isinstance(value, np.ndarray)
		assert np.issubdtype(value.dtype, np.bool_)
		assert (value.ndim == 2)
		assert (value.shape == (self.size_y, self.size_x))

		self._full_mask = value
	

	@property
	def visible_mask(self):
		return self._visible_mask
	
	@visible_mask.setter
	def visible_mask(self, value):
		assert isinstance(value, np.ndarray)
		assert np.issubdtype(value.dtype, np.bool_)
		assert (value.ndim == 2)
		assert (value.shape == (self.size_y, self.size_x))

		self._visible_mask = value
	

	@property
	def did_occlusion_object(self):
		return self._did_occlusion_object
	
	@did_occlusion_object.setter
	def did_occlusion_object(self, value):
		value = bool(value)
		self._did_occlusion_object = value
	

	@property
	def occ_obj_mask(self):
		return self._occ_obj_mask
	
	@occ_obj_mask.setter
	def occ_obj_mask(self, value):
		assert isinstance(value, np.ndarray)
		assert np.issubdtype(value.dtype, np.bool_)
		assert (value.ndim == 2)
		assert (value.shape == (self.size_y, self.size_x))

		self._occ_obj_mask = value
	

	@property
	def occ_obj_full_rectangle(self):
		return self._occ_obj_full_rectangle
	
	@occ_obj_full_rectangle.setter
	def occ_obj_full_rectangle(self, value):
		assert isinstance(value, util.geometry.Rectangle)
		self._occ_obj_full_rectangle = value


def _generate_occlusion_blob_middle_rectangle(primary_size_x, primary_size_y, expansion_margin_x, expansion_margin_y, config):
	expanded_size_x = primary_size_x + (2 * expansion_margin_x)
	expanded_size_y = primary_size_y + (2 * expansion_margin_y)

	back_rectangle = util.geometry.Rectangle(0.0, 0.0, 0.0, 0.0) # populated later

	# Blob middle occlusion
	occlusion_blob_middle_ratio = np.random.uniform(
		config.occlusion_blob_middle_min_ratio, config.occlusion_blob_middle_max_ratio, size=None
	)

	do_occlusion_blob_middle_vertical = bool(np.random.binomial(1, config.occlusion_blob_middle_verticle_prob, size=None))
	if (do_occlusion_blob_middle_vertical):
		# Blob middle vertical occlusion
		back_rectangle.size_x = (occlusion_blob_middle_ratio * primary_size_x)
		back_rectangle.min_x = np.random.uniform(
			float(expansion_margin_x), float(expansion_margin_x + primary_size_x - back_rectangle.size_x)
		)
		back_rectangle.size_y = expanded_size_y
		back_rectangle.min_y = 0.0
	else:
		# Blob middle horizontal occlusion
		back_rectangle.size_x = expanded_size_x
		back_rectangle.min_x = 0.0
		back_rectangle.size_y = (occlusion_blob_middle_ratio * primary_size_y)
		back_rectangle.min_y = np.random.uniform(
			float(expansion_margin_y), float(expansion_margin_y + primary_size_y - back_rectangle.size_y)
		)
	
	return back_rectangle


def _generate_occlusion_blob_side_rectangle(primary_size_x, primary_size_y, expansion_margin_x, expansion_margin_y, config):
	expanded_size_x = primary_size_x + (2 * expansion_margin_x)
	expanded_size_y = primary_size_y + (2 * expansion_margin_y)

	back_rectangle = util.geometry.Rectangle(0.0, 0.0, 0.0, 0.0) # populated later

	# Blob side occlusion
	occlusion_side_ratio = np.random.uniform(
		config.occlusion_blob_side_min_ratio, config.occlusion_blob_side_max_ratio
	)

	do_occlusion_blob_side_lr = bool(np.random.binomial(1, config.occlusion_blob_side_lr_prob, size=None))
	if do_occlusion_blob_side_lr:
		# Blob side left or right occlusion
		back_rectangle.size_x = (expansion_margin_x + (occlusion_side_ratio * primary_size_x))
		back_rectangle.size_y = expanded_size_y
		back_rectangle.min_y = 0.0

		do_occlusion_blob_side_left = bool(np.random.binomial(1, 0.5, size=None))
		if do_occlusion_blob_side_left:
			# Blob side left occlusion
			back_rectangle.min_x = 0.0
		else:
			# Blob side right occlusion
			back_rectangle.min_x = (expanded_size_x - back_rectangle.size_x)
	else:
		# Blob side top or bottom occlusion
		back_rectangle.size_x = expanded_size_x
		back_rectangle.min_x = 0.0
		back_rectangle.size_y = (expansion_margin_y + (occlusion_side_ratio * primary_size_y))

		do_occlusion_blob_side_top = bool(np.random.binomial(1, 0.5, size=None))
		if do_occlusion_blob_side_top:
			# Blob side top occlusion
			back_rectangle.min_y = 0.0
		else:
			# Blob side bottom occlusion
			back_rectangle.min_y = (expanded_size_y - back_rectangle.size_y)
	
	return back_rectangle


def _generate_occolusion_object_rectangles(primary_size_x, primary_size_y, expansion_margin_x, expansion_margin_y, 
		occ_obj_size_x, occ_obj_size_y, config):
	
	expanded_size_x = primary_size_x + (2 * expansion_margin_x)
	expanded_size_y = primary_size_y + (2 * expansion_margin_y)

	occlusion_object_ratio = np.random.uniform(
		config.occlusion_object_min_ratio, config.occlusion_object_max_ratio
	)
	occlusion_object_ratio_x = np.random.uniform(occlusion_object_ratio, 1.0)
	occlusion_object_ratio_y = occlusion_object_ratio / occlusion_object_ratio_x

	# This is the rectangle of the occluder object's image, in the reference frame of the expanded image.
	primary_occ_obj_image_rectangle = util.geometry.Rectangle(0.0, 0.0, occ_obj_size_x, occ_obj_size_y) # populated later

	# Handle different object occlusion types
	do_occlusion_object_left = bool(np.random.binomial(1, 0.5, size=None))
	if do_occlusion_object_left:
		# Left object occlusion
		primary_occ_obj_image_rectangle.min_x = (
			(expansion_margin_x + (occlusion_object_ratio_x * primary_size_x)) - primary_occ_obj_image_rectangle.size_x
		)
	else:
		# Right object occlusion
		primary_occ_obj_image_rectangle.min_x = (expansion_margin_x + ((1.0 - occlusion_object_ratio_x) * primary_size_x))

	do_occlusion_object_top = bool(np.random.binomial(1, 0.5, size=None))
	if do_occlusion_object_top:
		# Top object occlusion
		primary_occ_obj_image_rectangle.min_y = (
			(expansion_margin_y + (occlusion_object_ratio_y * primary_size_y)) - primary_occ_obj_image_rectangle.size_y
		)
	else:
		# Bottom object occlusion
		primary_occ_obj_image_rectangle.min_y = (expansion_margin_y + ((1.0 - occlusion_object_ratio_y) * primary_size_y))
	
	# The "back_rectangle" is the rectangle in expanded_rgba_image_float_array that will be occluded.
	back_rectangle = util.geometry.Rectangle(0.0, 0.0, 0.0, 0.0) # populated later
	back_rectangle.min_x = max(primary_occ_obj_image_rectangle.min_x, 0.0)
	back_rectangle.min_y = max(primary_occ_obj_image_rectangle.min_y, 0.0)
	back_rectangle.size_x = min(primary_occ_obj_image_rectangle.max_x, expanded_size_x) - back_rectangle.min_x
	back_rectangle.size_y = min(primary_occ_obj_image_rectangle.max_y, expanded_size_y) - back_rectangle.min_y

	# The "front_rectangle" is the rectangle in obj_occ_rgba_image_array that occludes.
	front_rectangle = back_rectangle.copy()
	front_rectangle.apply_translation(-primary_occ_obj_image_rectangle.min_x, -primary_occ_obj_image_rectangle.min_y)

	return (primary_occ_obj_image_rectangle, back_rectangle, front_rectangle)


def randomly_occlude_rgba(primary_image_array, occ_blob_image_array, occ_obj_rgba_image_array, config):
	assert isinstance(primary_image_array, np.ndarray)
	assert (primary_image_array.ndim >= 2)
	primary_size_y = primary_image_array.shape[0]
	primary_size_x = primary_image_array.shape[1]
	primary_rgba_image_float_array = image_to_rgba_float(primary_image_array)

	assert isinstance(occ_blob_image_array, np.ndarray)
	occ_blob_rgba_image_float_array = image_to_rgba_float(occ_blob_image_array)
	(occ_blob_size_y, occ_blob_size_x, _) = occ_blob_image_array.shape

	assert isinstance(occ_obj_rgba_image_array, np.ndarray)
	assert (occ_obj_rgba_image_array.ndim == 3)
	(occ_obj_size_y, occ_obj_size_x, occluder_num_channels) = occ_obj_rgba_image_array.shape
	assert (occluder_num_channels == 4)
	occ_obj_rgba_image_float_array = skimage.img_as_float(occ_obj_rgba_image_array)
	occ_obj_full_mask = (occ_obj_rgba_image_float_array[:, :, 3] >= 0.1)

	assert isinstance(config, RandomOcclusionConfig)

	# Squarize and expand primary image by expansion_ratio on all sides. 
	# This accomodates squarizing and adding perturbations to visible rectangle later.
	if (primary_size_x > primary_size_y):
		expansion_margin_x = int(np.ceil(config.expansion_ratio * primary_size_x))
		expansion_margin_y = expansion_margin_x + int(np.ceil(float(primary_size_x - primary_size_y) / 2.0))
	else:
		expansion_margin_y = int(np.ceil(config.expansion_ratio * primary_size_y))
		expansion_margin_x = expansion_margin_y + int(np.ceil(float(primary_size_y - primary_size_x) / 2.0))
	expanded_size_x = primary_size_x + (2 * expansion_margin_x)
	expanded_size_y = primary_size_y + (2 * expansion_margin_y)
	expanded_rgba_image_float_array = np.zeros((expanded_size_y, expanded_size_x, 4), dtype=primary_rgba_image_float_array.dtype)
	expanded_rgba_image_float_array[
		expansion_margin_y:(expansion_margin_y + primary_size_y), expansion_margin_x:(expansion_margin_x + primary_size_x), :
	] = primary_rgba_image_float_array

	# Initialize variables to return
	occluded_rgba_image_float_array = np.copy(expanded_rgba_image_float_array)
	primary_full_mask = (expanded_rgba_image_float_array[:, :, 3] >= 0.1)
	primary_visible_mask = np.copy(primary_full_mask)

	# Do occlusion?
	occlusion_result = RandomOcclusionResult(expanded_size_x, expanded_size_y)
	occlusion_result.did_occlusion_object = False
	do_occlusion = bool(np.random.binomial(1, config.occlusion_prob, size=None))
	if do_occlusion:

		do_occlusion_blob = bool(np.random.binomial(1, config.occlusion_blob_prob, size=None))
		if do_occlusion_blob:

			do_occlusion_blob_middle = bool(np.random.binomial(1, config.occlusion_blob_middle_prob, size=None))
			if do_occlusion_blob_middle:
				# Blob middle occlusion
				back_rectangle = _generate_occlusion_blob_middle_rectangle(
					primary_size_x, primary_size_y, expansion_margin_x, expansion_margin_y, config
				)
			else:
				# Blob side occlusion
				back_rectangle = _generate_occlusion_blob_side_rectangle(
					primary_size_x, primary_size_y, expansion_margin_x, expansion_margin_y, config
				)
			
			# Apply blob occlusion
			back_index_object = rectangle_to_image_index(back_rectangle)
			blob_size_y = back_index_object[0].stop - back_index_object[0].start
			blob_size_x = back_index_object[1].stop - back_index_object[1].start
			blob_max_size = max(blob_size_x, blob_size_y)
			if (min(occ_blob_size_x, occ_blob_size_y) < blob_max_size):
				occ_blob_rgba_image_float_array = scale_short_edge_to_length(occ_blob_rgba_image_float_array, blob_max_size)
			(front_patch, _, _) = randomly_crop_to_size(occ_blob_rgba_image_float_array, blob_size_x, blob_size_y)
			occluded_rgba_image_float_array[back_index_object] = front_patch
			primary_visible_mask[back_index_object] = False
		
		else:
			occlusion_result.did_occlusion_object = True
			(primary_occ_obj_image_rectangle, back_rectangle, front_rectangle) = _generate_occolusion_object_rectangles(
				primary_size_x, primary_size_y, expansion_margin_x, expansion_margin_y, occ_obj_size_x, occ_obj_size_y, config
			)
			back_index_object = rectangle_to_image_index(back_rectangle)
			front_index_object = rectangle_to_image_index(front_rectangle)

			# Apply object occlusion
			back_patch_float_array = expanded_rgba_image_float_array[back_index_object]
			front_patch_float_array = occ_obj_rgba_image_float_array[front_index_object]
			blended_patch_float_array = blend_rgba(front_patch_float_array, back_patch_float_array)
			occluded_rgba_image_float_array[back_index_object] = blended_patch_float_array

			front_patch_visible_mask = occ_obj_full_mask[front_index_object]
			primary_visible_mask[back_index_object] = np.logical_and(
				primary_visible_mask[back_index_object], np.logical_not(front_patch_visible_mask)
			)

			primary_occ_obj_mask = np.zeros((expanded_size_y, expanded_size_x), dtype=np.bool_)
			primary_occ_obj_mask[back_index_object] = front_patch_visible_mask
			occlusion_result.occ_obj_mask = primary_occ_obj_mask

			occ_obj_image_rectangle = util.geometry.Rectangle(0.0, 0.0, occ_obj_size_x, occ_obj_size_y)
			occ_obj_full_rectangle = util.geometry.nonzero_bounding_rectangle(occ_obj_full_mask)
			primary_occ_obj_full_rectangle = primary_occ_obj_image_rectangle.copy()
			primary_occ_obj_full_rectangle.apply_shift_param(
				occ_obj_image_rectangle.compute_shift_param_to_rectangle(occ_obj_full_rectangle)
			)
			occlusion_result.occ_obj_full_rectangle = primary_occ_obj_full_rectangle
	
	occlusion_result.rgba_image_float_array = occluded_rgba_image_float_array
	occlusion_result.full_mask = primary_full_mask
	occlusion_result.visible_mask = primary_visible_mask
	
	return occlusion_result


def compute_hog_feature_length(image_size_x, image_size_y, num_orientations, cell_size_x, cell_size_y, block_size_x, block_size_y):
	"""
	References:
		Histograms of Oriented Gradients for Human Detection, Dalal, N and Triggs, B, 2005
		https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_hog.py#L25
	"""
	image_size_x = int(image_size_x)
	assert (image_size_x > 0)
	image_size_y = int(image_size_y)
	assert (image_size_y > 0)

	num_orientations = int(num_orientations)
	assert (num_orientations > 0)

	cell_size_x = int(cell_size_x)
	assert (cell_size_x > 0)
	cell_size_y = int(cell_size_y)
	assert (cell_size_y > 0)

	block_size_x = int(block_size_x)
	assert (block_size_x > 0)
	block_size_y = int(block_size_y)
	assert (block_size_y > 0)

	num_cells_x = image_size_x // cell_size_x
	num_cells_y = image_size_y // cell_size_y

	num_blocks_x = (num_cells_x - block_size_x) + 1
	num_blocks_y = (num_cells_y - block_size_y) + 1

	hog_feature_length = (num_blocks_x * num_blocks_y * block_size_x * block_size_y * num_orientations)
	return hog_feature_length


def pyramid_hog(image_array, num_layers, downscale_factor, num_orientations, cell_size_x, cell_size_y, 
    				block_size_x, block_size_y, block_norm, return_vector):
	"""
	Compute the HOG descriptor for the pyramid of an image.
	This is basically a wrapper for skimage.transform.pyramid_gaussian and skimage.feature.hog

	Inputs:
		image_array: An [image_size_y, image_size_x, num_channels] integer or float array. 
		             If this is not grayscale, will be converted to grayscale before computing HOG, since
	                 skimage.feature.hog only supports grayscale images.
		
		num_layers      : Integer indicating number of layers in the pyramid.
		downscale_factor: Float indicating the downscale factor between two layers.

		num_orientations:
		    ...         : Will be passed to skimage.feature.hog
		return_vector   :  
	
	Outputs:
		hog_feature_list: A Python list of features returned by skimage.feature.hog, one for each layer in the pyramid. 

	References:
		http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.pyramid_gaussian
		http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog
	"""
	# Check arguments
	assert isinstance(image_array, np.ndarray)
	assert ((image_array.ndim == 2) or (image_array.ndim == 3))
	if (image_array.ndim == 3):
		assert ((image_array.shape[2] == 3) or (image_array.shape[2] == 4))
	image_float_array = skimage.img_as_float(image_array)
	gray_image_float_array = skimage.color.rgb2gray(image_float_array)

	num_layers = int(num_layers)
	assert (num_layers >= 1)

	downscale_factor = float(downscale_factor)
	assert (downscale_factor > 1.0)

	# Other arguments are directly passed to skimage.feature.hog without checking

	# Construct image pyramid
	image_pyramid = skimage.transform.pyramid_gaussian(
		gray_image_float_array, max_layer=(num_layers-1), downscale=downscale_factor
	)

	# Compute HOG features
	hog_feature_list = []
	for layer_image_array in image_pyramid:
		# skimage.feature.hog will throw an exception when image size is too small.
		# Specifically, this happens when ((image_size // cell_size) - block_size + 1) is negative for x or y.
		# In other words, HOG does not know what to do when "the number of complete blocks"" is negative.
		# print(layer_image_array.shape)
		(layer_image_size_y, layer_image_size_x) = layer_image_array.shape
		if ((((layer_image_size_x // cell_size_x) - block_size_x) < -1) or (((layer_image_size_y // cell_size_y) - block_size_y) < -1)):
			break
		
		# skimage.feature.hog will thrown another exception if any image dimension happens to be 1 (not enough to compute gradient).
		# We want to avoid this situation as well.
		if ((layer_image_size_x <= 1) or (layer_image_size_y <= 1)):
			break

		(hog_feature, hog_image) = skimage.feature.hog(
			layer_image_array, orientations=num_orientations, pixels_per_cell=(cell_size_x, cell_size_y), 
			cells_per_block=(block_size_x, block_size_y), block_norm=block_norm, visualise=True, feature_vector=return_vector
		)

		# axes1 = matplotlib.pyplot.subplot(1, 2, 1)
		# axes1.imshow(layer_image_array, cmap="gray")
		# axes2 = matplotlib.pyplot.subplot(1, 2, 2)
		# axes2.imshow(hog_image, cmap="gray")
		# matplotlib.pyplot.show()

		hog_feature_list.append(hog_feature)
	
	return hog_feature_list


def compute_pyramid_hog_feature_length(image_size_x, image_size_y, num_layers, downscale_factor, 
					num_orientations, cell_size_x, cell_size_y, block_size_x, block_size_y):
	"""
	References:
		http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.pyramid_gaussian
		http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog
	"""
	image_size_x = int(image_size_x)
	assert (image_size_x > 0)
	image_size_y = int(image_size_y)
	assert (image_size_y > 0)

	num_layers = int(num_layers)
	assert (num_layers >= 1)

	downscale_factor = float(downscale_factor)
	assert (downscale_factor > 1.0)

	num_orientations = int(num_orientations)
	assert (num_orientations > 0)

	cell_size_x = int(cell_size_x)
	assert (cell_size_x > 0)
	cell_size_y = int(cell_size_y)
	assert (cell_size_y > 0)

	block_size_x = int(block_size_x)
	assert (block_size_x > 0)
	block_size_y = int(block_size_y)
	assert (block_size_y > 0)


	hog_feature_length_list = []
	hog_feature_length_list.append(compute_hog_feature_length(
		image_size_x=image_size_x, image_size_y=image_size_y, num_orientations=num_orientations, 
		cell_size_x=cell_size_x, cell_size_y=cell_size_y, block_size_x=block_size_x, block_size_y=block_size_y
	))

	prev_layer_image_size_x = image_size_x
	prev_layer_image_size_y = image_size_y
	for layer_index in range(1, num_layers):
		layer_image_size_x = int(np.ceil(float(prev_layer_image_size_x) / downscale_factor))
		layer_image_size_y = int(np.ceil(float(prev_layer_image_size_y) / downscale_factor))
		# print(layer_image_size_y, layer_image_size_x)
		
		# This break corresponds to the break in skimage.transform.pyramid_gaussian()
		if ((layer_image_size_x == prev_layer_image_size_x) and (layer_image_size_y == prev_layer_image_size_y)):
			break

		# These breaks corresponds to the break in pyramid_hog()
		if ((((layer_image_size_x // cell_size_x) - block_size_x) < -1) or (((layer_image_size_y // cell_size_y) - block_size_y) < -1)):
			break
		if ((layer_image_size_x <= 1) or (layer_image_size_y <= 1)):
			break
		
		hog_feature_length_list.append(compute_hog_feature_length(
			image_size_x=layer_image_size_x, image_size_y=layer_image_size_y, num_orientations=num_orientations, 
			cell_size_x=cell_size_x, cell_size_y=cell_size_y, block_size_x=block_size_x, block_size_y=block_size_y
		))
		
		prev_layer_image_size_x = layer_image_size_x
		prev_layer_image_size_y = layer_image_size_y
	
	return np.array(hog_feature_length_list, dtype=np.int32)