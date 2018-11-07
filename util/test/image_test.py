from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import skimage.color
import skimage.feature

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, "../.."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import util.image


def rgba_to_gray_test1():
	num_samples = 10

	for sample_index in range(num_samples):
		image_size_x = np.random.randint(160, 320 + 1)
		image_size_y = np.random.randint(120, 240 + 1)

		rgba_image_int_array = (np.random.random((image_size_y, image_size_x, 4)) * 255.0).astype(np.uint8)
		rgba_image_float_array = skimage.img_as_float(rgba_image_int_array)

		background_intensity = np.random.random()

		gt_gray_image_float_array = (
			skimage.color.rgb2gray(rgba_image_float_array[:, :, 0:3]) * rgba_image_float_array[:, :, 3] + 
			(background_intensity * (1.0 - rgba_image_float_array[:, :, 3]))
		)

		# test integer input
		test_gray_image_float_array = util.image.rgba_to_gray(rgba_image_int_array, background_intensity)
		assert np.all((np.abs(gt_gray_image_float_array - test_gray_image_float_array) < 1e-3), axis=None, keepdims=False)

		# test float input
		test_gray_image_float_array = util.image.rgba_to_gray(rgba_image_float_array, background_intensity)
		assert np.all((np.abs(gt_gray_image_float_array - test_gray_image_float_array) < 1e-3), axis=None, keepdims=False)


def compute_hog_feature_length_test1():
	num_samples = 10

	for sample_index in range(num_samples):
		num_orientations = np.random.randint(4, 18 + 1)
		image_size_x = np.random.randint(160, 320 + 1)
		image_size_y = np.random.randint(120, 240 + 1)
		cell_size_x = np.random.randint(4, 12 + 1)
		cell_size_y = np.random.randint(4, 12 + 1)
		block_size_x = np.random.randint(1, 3 + 1)
		block_size_y = np.random.randint(1, 3 + 1)

		gray_image_float_array = np.random.random((image_size_y, image_size_x))
		hog_feature = skimage.feature.hog(
			gray_image_float_array, orientations=num_orientations, pixels_per_cell=(cell_size_x, cell_size_y), 
			cells_per_block=(block_size_x, block_size_y), block_norm="L2", visualise=False, feature_vector=True
		)
		gt_hog_feature_length = hog_feature.size

		test_hog_feature_length = util.image.compute_hog_feature_length(
			image_size_x=image_size_x, image_size_y=image_size_y, num_orientations=num_orientations, 
			cell_size_x=cell_size_x, cell_size_y=cell_size_y, block_size_x=block_size_x, block_size_y=block_size_y
		)

		assert (test_hog_feature_length == gt_hog_feature_length)


def compute_pyramid_hog_feature_length_test1():
	num_samples = 100

	for sample_index in range(num_samples):
		num_orientations = np.random.randint(4, 18 + 1)
		image_size_x = np.random.randint(160, 320 + 1)
		image_size_y = np.random.randint(120, 240 + 1)
		cell_size_x = np.random.randint(4, 12 + 1)
		cell_size_y = np.random.randint(4, 12 + 1)
		block_size_x = np.random.randint(1, 3 + 1)
		block_size_y = np.random.randint(1, 3 + 1)

		gray_image_float_array = np.random.random((image_size_y, image_size_x))

		num_layers = np.random.randint(1, 20 + 1)
		downscale_factor = (2.0 * np.random.random()) + 1.0

		hog_feature_list = util.image.pyramid_hog(
			gray_image_float_array, num_layers=num_layers, downscale_factor=downscale_factor, 
			num_orientations=num_orientations, cell_size_x=cell_size_x, cell_size_y=cell_size_y, 
			block_size_x=block_size_x, block_size_y=block_size_y, block_norm="L2", return_vector=True
		)
		gt_hog_feature_length_list = [hog_feature.size for hog_feature in hog_feature_list]


		test_hog_feature_length_list = util.image.compute_pyramid_hog_feature_length(
			image_size_x=image_size_x, image_size_y=image_size_y, num_layers=num_layers, downscale_factor=downscale_factor, 
			num_orientations=num_orientations, cell_size_x=cell_size_x, cell_size_y=cell_size_y, block_size_x=block_size_x, block_size_y=block_size_y
		)

		assert (len(test_hog_feature_length_list) == len(gt_hog_feature_length_list))
		for (test_hog_feature_length, gt_hog_feature_length) in zip(test_hog_feature_length_list, gt_hog_feature_length_list):
			assert (test_hog_feature_length == gt_hog_feature_length)
		
		print("Test example %d out of %d" % (sample_index, num_samples))