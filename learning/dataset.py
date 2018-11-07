from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import torch.utils.data
import skimage.transform

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, ".."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import global_variables as global_vars
import util.geometry
import util.image
import data.object_class
import data.pytorch_wrapper


class ViewpointDataset(torch.utils.data.Dataset):
	"""
	The dataset used to train a viewpoint network.
	"""

	def __init__(self, augmented_rs_dataset, max_perturb_ratio):
		assert isinstance(augmented_rs_dataset, data.pytorch_wrapper.ShapenetRSAugmentedDataset)
		self._augmented_rs_dataset= augmented_rs_dataset

		max_perturb_ratio = float(max_perturb_ratio)
		assert (max_perturb_ratio >= 0.0)
		assert (max_perturb_ratio < 0.5)
		assert (max_perturb_ratio <= self._augmented_rs_dataset.expansion_ratio)
		self._max_perturb_ratio = max_perturb_ratio
	

	@property
	def shapenet_synset(self):
		return self._augmented_rs_dataset.shapenet_synset
	

	@property
	def class_index(self):
		return self._augmented_rs_dataset.class_index
	

	@property
	def max_perturb_ratio(self):
		return self._max_perturb_ratio
	

	def __len__(self):
		return len(self._augmented_rs_dataset)
	

	def get_item_with_extra(self, item_index):
		# Get item from AugmentedRSDataset
		augmented_datum = self._augmented_rs_dataset[item_index]
		occlusion_result = augmented_datum.occlusion_result

		# Compute full and visible rectangles
		full_rectangle = util.geometry.nonzero_bounding_rectangle(occlusion_result.full_mask)
		if (not np.any(occlusion_result.visible_mask, axis=None, keepdims=False)):
			print("Warining: in ViewpointDataset, no visible pixels! Using full rectangle instead!")
			visible_rectangle = full_rectangle.copy()
		else:
			visible_rectangle = util.geometry.nonzero_bounding_rectangle(occlusion_result.visible_mask)
			if ((visible_rectangle.size_x < 5.0) or (visible_rectangle.size_y < 5.0)):
				print("Warning: in ViewpointDataset, visible rectangle %s is too small! Using full rectange instead!" % (visible_rectangle, ))
				visible_rectangle = full_rectangle.copy()
		
		# Construct crop rectangle. Randomly perturb each side by a certain value
		perturb_ratio_vector = np.random.uniform(-self.max_perturb_ratio, self.max_perturb_ratio, size=[4])
		crop_rectangle = visible_rectangle.copy()
		crop_rectangle.apply_shift_ratios(*perturb_ratio_vector)

		# Crop and resize to input image.
		crop_index_object = util.image.rectangle_to_image_index(crop_rectangle)
		cropped_rgb_image_float_array = augmented_datum.augmented_rgb_image_float_array[crop_index_object]
		input_rgb_image_float_array = skimage.transform.resize(
			cropped_rgb_image_float_array, (global_vars.CNN_VIEWPOINT_INPUT_SIZE, global_vars.CNN_VIEWPOINT_INPUT_SIZE), 
			order=1, mode="constant", preserve_range=False, anti_aliasing=True
		)

		# Construct outputs
		input_rgb_image_float_tensor = torch.from_numpy(np.transpose(input_rgb_image_float_array, (2, 0, 1))).float()
		viewpoint_tensor = torch.FloatTensor(augmented_datum.viewpoint_tuple)
		extra_item_tuple = (augmented_datum, crop_rectangle)

		return (input_rgb_image_float_tensor, viewpoint_tensor, extra_item_tuple)
	

	def __getitem__(self, item_index):
		(
			input_rgb_image_float_tensor, viewpoint_tensor, _
		) = self.get_item_with_extra(item_index)

		return (input_rgb_image_float_tensor, viewpoint_tensor)


# Visualization code to be inserted after computing rectangles:

		# print(full_rectangle)
		# print(visible_rectangle)
		# print(crop_rectangle)

		# import matplotlib.pyplot
		# visualization_figure = matplotlib.pyplot.figure()
		# visualization_axes = visualization_figure.add_subplot(1, 1, 1)
		# visualization_axes.imshow(augmented_datum.augmented_rgb_image_float_array)
		# full_rectangle_patch = matplotlib.patches.Rectangle(
		# 	(full_rectangle.min_x, full_rectangle.min_y), full_rectangle.size_x, full_rectangle.size_y, linewidth=1, edgecolor="k", fill=False
		# )
		# visualization_axes.add_patch(full_rectangle_patch)
		# visible_rectangle_patch = matplotlib.patches.Rectangle(
		# 	(visible_rectangle.min_x, visible_rectangle.min_y), visible_rectangle.size_x, visible_rectangle.size_y, linewidth=1, edgecolor="b", fill=False
		# )
		# visualization_axes.add_patch(visible_rectangle_patch)
		# crop_rectangle_patch = matplotlib.patches.Rectangle(
		# 	(crop_rectangle.min_x, crop_rectangle.min_y), crop_rectangle.size_x, crop_rectangle.size_y, linewidth=1, edgecolor="r", fill=False
		# )
		# visualization_axes.add_patch(crop_rectangle_patch)
		# matplotlib.pyplot.show()