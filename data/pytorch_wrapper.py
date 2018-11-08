from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import io
import json
import numpy as np
import torch.utils.data
import skimage.io
import skimage.util

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, ".."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import global_variables as global_vars
import util.image
import data.object_class
import data.shapenet.meta_pb2 # If this is not found, bash ../../build_protobuf.sh


class ImageDirDataset(torch.utils.data.Dataset):
	"""
	Wrapper for a single directory containing multiple images.

	Arguments:
		image_dir (str): The path of the random synthesis metadata file.
	"""

	def __init__(self, image_dir, do_image_caching):
		# Check arguments
		image_dir = str(image_dir)
		assert os.path.isdir(image_dir)
		self._image_dir = image_dir

		do_image_caching = bool(do_image_caching)
		self._do_image_caching = do_image_caching

		# Main loop
		image_dir_entry_list = os.listdir(self.image_dir)
		image_file_name_list = []
		for image_dir_entry in image_dir_entry_list:
			if ((not image_dir_entry.endswith(".jpg")) and (not image_dir_entry.endswith(".png"))):
				continue
			
			image_path = os.path.join(self.image_dir, image_dir_entry)
			if (not os.path.isfile(image_path)):
				continue
			
			image_file_name_list.append(image_dir_entry)
		
		# initialize states
		self._image_file_name_list = image_file_name_list
		if self.do_image_caching:
			self._image_content_list = np.empty([len(image_file_name_list)], dtype=object)
		

	@property
	def image_dir(self):
		return self._image_dir
	

	@property
	def do_image_caching(self):
		return self._do_image_caching
	

	def __len__(self):
		return len(self._image_file_name_list)
	

	def __getitem__(self, item_index):
		item_index = int(item_index)
		assert (item_index >= 0)
		assert (item_index < len(self))

		image_file_name = self._image_file_name_list[item_index]
		image_file_path = os.path.join(self.image_dir, image_file_name)
		if self.do_image_caching: 
			if (self._image_content_list[item_index] is None):
				image_file = open(image_file_path, "rb")
				self._image_content_list[item_index] = image_file.read()
				image_file.close()
			
			image_content_binary_io = io.BytesIO(self._image_content_list[item_index])
			image_ubyte_array = skimage.io.imread(image_content_binary_io)
			image_content_binary_io.close()
		else:
			image_ubyte_array = skimage.io.imread(image_file_path)

		return image_ubyte_array


class ShapenetRSDataset(torch.utils.data.Dataset):
	"""
	Wrapper for random synthesis data of a ShapeNet synset.

	Arguments:
		rs_meta_file_path (str): The path of the random synthesis metadata file.
	"""

	def __init__(self, rs_meta_file_path):
		# Check arguments
		rs_meta_file_path = str(rs_meta_file_path)
		assert os.path.isfile(rs_meta_file_path)
		rs_synset_dir = os.path.dirname(rs_meta_file_path)
		self._rs_synset_dir = rs_synset_dir

		# Read random synthesis meta file
		rs_meta_pbobj = data.shapenet.meta_pb2.RSSynsetMeta()
		rs_meta_file = open(rs_meta_file_path, "rb")
		rs_meta_pbobj.ParseFromString(rs_meta_file.read())
		rs_meta_file.close()

		# Process random synthesis meta file
		shapenet_synset = rs_meta_pbobj.header.shapenet_synset
		assert (len(shapenet_synset) > 0)
		max_num_configs = rs_meta_pbobj.header.num_configs
		assert (max_num_configs > 0)
		num_models = len(rs_meta_pbobj.model_meta_list)

		model_shapenet_id_list = []
		item_shapenet_id_list = []
		image_relative_path_list = []
		viewpoint_tuple_list = []
		for model_index in range(num_models):
			model_meta_pbobj = rs_meta_pbobj.model_meta_list[model_index]
			model_shapenet_id_list.append(model_meta_pbobj.shapenet_id)

			num_configs = len(model_meta_pbobj.record_meta_list)
			assert (num_configs <= max_num_configs)
			for config_index in range(num_configs):
				record_meta_pbobj = model_meta_pbobj.record_meta_list[config_index]

				item_shapenet_id_list.append(model_meta_pbobj.shapenet_id)
				image_relative_path_list.append(record_meta_pbobj.image_relative_path)
				viewpoint_tuple_list.append((
					record_meta_pbobj.camera_pose.azimuth_rad, 
					record_meta_pbobj.camera_pose.elevation_rad, 
					record_meta_pbobj.camera_pose.axial_rotation_rad 
				))

		# Store in state
		self._shapenet_synset = shapenet_synset
		self._model_shapenet_id_list = model_shapenet_id_list
		self._item_shapenet_id_list = item_shapenet_id_list
		self._image_relative_path_list = image_relative_path_list
		self._viewpoint_tuple_list = viewpoint_tuple_list
	

	@property
	def shapenet_synset(self):
		return self._shapenet_synset
	

	@property
	def class_index(self):
		return int(data.object_class.shapenet_synset_to_main_index(self.shapenet_synset))
	

	@property
	def rs_synset_dir(self):
		return self._rs_synset_dir
	

	@property
	def model_shapenet_id_list(self):
		return self._model_shapenet_id_list
	

	def __len__(self):
		return len(self._item_shapenet_id_list)
	

	def __getitem__(self, item_index):
		item_index = int(item_index)
		assert (item_index >= 0)
		assert (item_index < len(self))

		item_shapenet_id = self._item_shapenet_id_list[item_index]

		image_relative_path = self._image_relative_path_list[item_index]
		image_path = os.path.join(self.rs_synset_dir, image_relative_path)
		rgba_image_ubyte_array = skimage.io.imread(image_path)
		assert (rgba_image_ubyte_array.ndim == 3)
		assert (rgba_image_ubyte_array.shape[2] == 4)

		viewpoint_tuple = self._viewpoint_tuple_list[item_index]

		return (item_shapenet_id, rgba_image_ubyte_array, viewpoint_tuple)


class ShapenetRSAugmentedDatum(object):
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
	def primary_shapenet_id(self):
		return self._primary_shapenet_id
	
	@primary_shapenet_id.setter
	def primary_shapenet_id(self, value):
		value = str(value)
		assert (len(value) > 0)

		self._primary_shapenet_id = value
	

	@property
	def augmented_rgb_image_float_array(self):
		return self._augmented_rgb_image_float_array
	
	@augmented_rgb_image_float_array.setter
	def augmented_rgb_image_float_array(self, value):
		assert isinstance(value, np.ndarray)
		assert np.issubdtype(value.dtype, np.floating)
		assert (value.ndim == 3)
		assert (value.shape == (self.size_y, self.size_x, 3))

		self._augmented_rgb_image_float_array = value
	

	@property
	def occlusion_result(self):
		return self._occlusion_result
	
	@occlusion_result.setter
	def occlusion_result(self, value):
		assert isinstance(value, util.image.RandomOcclusionResult)
		assert (value.size_x == self.size_x)
		assert (value.size_y == self.size_y)

		self._occlusion_result = value
	

	@property
	def occ_obj_class_index(self):
		return self._occ_obj_class_index
	
	@occ_obj_class_index.setter
	def occ_obj_class_index(self, value):
		value = int(value)
		assert (value >= 0)
		assert (value < data.object_class.num_main_classes())

		self._occ_obj_class_index = value
	

	@property
	def viewpoint_tuple(self):
		return self._viewpoint_tuple
	
	@viewpoint_tuple.setter
	def viewpoint_tuple(self, value):
		assert isinstance(value, tuple)
		assert (len(value) == 3)

		self._viewpoint_tuple = value



class ShapenetRSAugmentedDataset(torch.utils.data.Dataset):
	"""
	Wrapper for random synthesis data of a ShapeNet synset, with numerous data augmentations:
		- Random occlusions
		- Real image background

	Arguments:
		primary_dataset (ShapenetRSDataset): The dataset containing the primary objects.
		occ_obj_dataset_list (list<ShapenetRSDataset>): A list of datasets containing occluder objects.
		background_dataset (torch.utils.data.Dataset): The dataset containing background images.
	"""

	def __init__(self, primary_dataset, occ_obj_dataset_list, background_dataset, occlusion_config):
		# Check arguments
		assert isinstance(primary_dataset, ShapenetRSDataset)
		assert (len(primary_dataset) > 0)

		assert (len(occ_obj_dataset_list) > 0)
		for occ_obj_dataset in occ_obj_dataset_list:
			assert isinstance(occ_obj_dataset, ShapenetRSDataset)
			assert (len(occ_obj_dataset) > 0)
		
		assert isinstance(background_dataset, torch.utils.data.Dataset)
		assert (len(background_dataset) > 0)

		assert isinstance(occlusion_config, util.image.RandomOcclusionConfig)

		# Save to states
		self._primary_dataset = primary_dataset
		self._occ_obj_dataset_list = list(occ_obj_dataset_list)
		self._background_dataset = background_dataset
		self._occlusion_config = occlusion_config
	

	@property
	def shapenet_synset(self):
		return self._primary_dataset.shapenet_synset
	

	@property
	def class_index(self):
		return self._primary_dataset.class_index
	

	@property
	def model_shapenet_id_list(self):
		return self._primary_dataset.model_shapenet_id_list
	

	@property
	def expansion_ratio(self):
		return self._occlusion_config.expansion_ratio
	

	def __len__(self):
		return len(self._primary_dataset)
	

	def __getitem__(self, item_index):
		# Obtain primary data
		(
			primary_shapenet_id, primary_rgba_image_ubyte_array, primary_viewpoint_tuple
		) = self._primary_dataset[item_index]
		primary_rgba_image_ubyte_array = util.image.cut_rgba_transparent_border(primary_rgba_image_ubyte_array)

		# Obtain blob occluder image.
		occ_blob_index = np.random.randint(0, len(self._background_dataset), size=None)
		occ_blob_image_ubyte_array = self._background_dataset[occ_blob_index]
		occ_blob_rgb_image_float_array = util.image.image_to_rgb_float(occ_blob_image_ubyte_array)

		# Obtain object occluder image.
		occ_obj_dataset_index = np.random.randint(0, len(self._occ_obj_dataset_list), size=None)
		occ_obj_dataset = self._occ_obj_dataset_list[occ_obj_dataset_index]
		occ_obj_index = np.random.randint(0, len(occ_obj_dataset), size=None)
		(_, occ_obj_rgba_image_ubyte_array, _) = occ_obj_dataset[occ_obj_index]
		occ_obj_rgba_image_ubyte_array = util.image.cut_rgba_transparent_border(occ_obj_rgba_image_ubyte_array)

		# Apply random occlusion
		occlusion_result = util.image.randomly_occlude_rgba(
			primary_rgba_image_ubyte_array, occ_blob_rgb_image_float_array, occ_obj_rgba_image_ubyte_array, self._occlusion_config
		)
		occ_result_max_size = max(occlusion_result.size_x, occlusion_result.size_y)

		# Get background image and patch
		background_index = np.random.randint(0, len(self._background_dataset), size=None)
		bg_image_ubyte_array = self._background_dataset[background_index]
		bg_rgb_image_float_array = util.image.image_to_rgb_float(bg_image_ubyte_array)
		(bg_size_y, bg_size_x, _) = bg_rgb_image_float_array.shape
		if (min(bg_size_x, bg_size_y) < occ_result_max_size):
			bg_rgb_image_float_array = util.image.scale_short_edge_to_length(bg_rgb_image_float_array, occ_result_max_size)
		(bg_rgb_patch_float_array, _, _) = util.image.randomly_crop_to_size(bg_rgb_image_float_array, occlusion_result.size_x, occlusion_result.size_y)

		# Overlay foreground image and background image
		foreground_weight_matrix = occlusion_result.rgba_image_float_array[:, :, 3, np.newaxis]
		augmented_rgb_image_float_array = (
			(occlusion_result.rgba_image_float_array[:, :, 0:3] * foreground_weight_matrix) + 
			(bg_rgb_patch_float_array * (1.0 - foreground_weight_matrix))
		)

		# Add some noise to the augmented image
		# augmented_rgb_image_float_array = skimage.util.random_noise(
		# 	augmented_rgb_image_float_array, mode="gaussian", clip=True, var=0.005
		# )

		# Construct datum to return
		augmented_datum = ShapenetRSAugmentedDatum(occlusion_result.size_x, occlusion_result.size_y)
		augmented_datum.primary_shapenet_id = primary_shapenet_id
		augmented_datum.augmented_rgb_image_float_array = augmented_rgb_image_float_array
		augmented_datum.occlusion_result = occlusion_result
		augmented_datum.occ_obj_class_index = occ_obj_dataset.class_index
		augmented_datum.viewpoint_tuple = primary_viewpoint_tuple

		return augmented_datum


def construct_rs_augmented_dataset(primary_class_enum, background_dataset, occlusion_config):
	assert isinstance(primary_class_enum, data.object_class.MainClass)
	primary_class_index = data.object_class.main_enum_to_main_index(primary_class_enum)
	assert isinstance(background_dataset, torch.utils.data.Dataset)
	assert isinstance(occlusion_config, util.image.RandomOcclusionConfig)

	class_index_list = range(data.object_class.num_main_classes())
	shapenet_synset_list = data.object_class.main_index_to_shapenet_synset(class_index_list)
	rs_meta_file_path_list = [
		os.path.join(global_vars.SHAPENET_RENDER_DIR, shapenet_synset, global_vars.SHAPENET_META_FILE_NAME) 
		for shapenet_synset in shapenet_synset_list
	]

	primary_rs_meta_file_path = rs_meta_file_path_list[primary_class_index]
	primary_rs_dataset = ShapenetRSDataset(primary_rs_meta_file_path)

	occluder_class_index_vector = data.object_class.get_occluder_main_index_vector(primary_class_index)
	occ_obj_dataset_list = []
	for occluder_class_index in occluder_class_index_vector:
		occluder_rs_meta_file_path = rs_meta_file_path_list[occluder_class_index]
		occ_obj_dataset = ShapenetRSDataset(occluder_rs_meta_file_path)
		occ_obj_dataset_list.append(occ_obj_dataset)

	rs_augmented_dataset = ShapenetRSAugmentedDataset(
		primary_rs_dataset, occ_obj_dataset_list, background_dataset, occlusion_config
	)
	return rs_augmented_dataset


# Code for testing ShapenetRSAugmentedDataset

# if __name__ == "__main__":
# 	import matplotlib.pyplot
# 	import util.geometry

# 	places2_val_dataset = ImageDirDataset(global_vars.PLACES2_VAL_DIR, False)
# 	occlusion_config = util.image.RandomOcclusionConfig()
# 	occlusion_config.expansion_ratio = 0.2
# 	occlusion_config.occlusion_blob_middle_verticle_prob = 1.0
# 	occlusion_config.occlusion_blob_side_lr_prob = 1.0
# 	car_rs_augmented_dataset = construct_rs_augmented_dataset(
# 		data.object_class.MainClass.CAR, places2_val_dataset, occlusion_config
# 	)

# 	for dummy_index in range(100):
# 		example_index = np.random.randint(0, len(car_rs_augmented_dataset), size=None)
# 		augmented_datum = car_rs_augmented_dataset[example_index]
# 		occlusion_result = augmented_datum.occlusion_result

# 		visualization_figure = matplotlib.pyplot.figure()

# 		augmented_image_axes = visualization_figure.add_subplot(1, 1, 1)
# 		augmented_image_axes.imshow(augmented_datum.augmented_rgb_image_float_array)
# 		full_rectangle = util.geometry.nonzero_bounding_rectangle(occlusion_result.full_mask)
# 		full_rectangle_patch = matplotlib.patches.Rectangle(
# 			(full_rectangle.min_x, full_rectangle.min_y), full_rectangle.size_x, full_rectangle.size_y, 
# 			linewidth=3, edgecolor="g", linestyle="--", fill=False
# 		)
# 		augmented_image_axes.add_patch(full_rectangle_patch)
# 		visible_rectangle = util.geometry.nonzero_bounding_rectangle(occlusion_result.visible_mask)
# 		visible_rectangle_patch = matplotlib.patches.Rectangle(
# 			(visible_rectangle.min_x, visible_rectangle.min_y), visible_rectangle.size_x, visible_rectangle.size_y, 
# 			linewidth=1, edgecolor="r", linestyle="-", fill=False
# 		)
# 		augmented_image_axes.add_patch(visible_rectangle_patch)
# 		augmented_image_axes.legend(["Full BBox", "Visible BBox"])
# 		augmented_image_axes.axis("off")

# 		# full_mask_axes = visualization_figure.add_subplot(2, 2, 2)
# 		# full_mask_axes.imshow(occlusion_result.full_mask, cmap="gray")

# 		# visible_mask_axes = visualization_figure.add_subplot(2, 2, 3)
# 		# visible_mask_axes.imshow(occlusion_result.visible_mask, cmap="gray")

# 		# if (occlusion_result.did_occlusion_object):
# 		# 	occ_obj_mask_axes = visualization_figure.add_subplot(2, 2, 4)
# 		# 	occ_obj_mask_axes.imshow(occlusion_result.occ_obj_mask, cmap="gray")
# 		# 	occ_obj_full_rectangle = occlusion_result.occ_obj_full_rectangle
# 		# 	occ_obj_full_rectangle_patch = matplotlib.patches.Rectangle(
# 		# 		(occ_obj_full_rectangle.min_x, occ_obj_full_rectangle.min_y), occ_obj_full_rectangle.size_x, occ_obj_full_rectangle.size_y, 
# 		# 		linewidth=1, edgecolor="y", fill=False
# 		# 	)
# 		# 	augmented_image_axes.add_patch(occ_obj_full_rectangle_patch)


# 		print(augmented_datum.viewpoint_tuple)
# 		matplotlib.pyplot.show()