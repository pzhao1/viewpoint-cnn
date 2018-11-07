from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import numpy as np

#####################
#       MAIN        #
#####################
# This is the class types that will be used in the networks / scrips of this project.
# All other class types will be converted to this form.

@enum.unique
class MainClass(enum.IntEnum):
	CHAIR = 0
	TABLE = 1
	CAR = 2


def num_main_classes():
	return len(MainClass)


_main_index_to_enum_vector = np.empty([num_main_classes()], dtype=object)
for _main_index in range(num_main_classes()):
	_main_index_to_enum_vector[_main_index] = MainClass(_main_index)
del _main_index


def main_index_to_main_enum(input_main_index_maybe_array):
	input_main_index_array = np.array(input_main_index_maybe_array, dtype=np.int32)
	assert np.all((input_main_index_array >= 0), axis=None, keepdims=False)
	assert np.all((input_main_index_array < num_main_classes()), axis=None, keepdims=False)

	output_main_enum_maybe_array = _main_index_to_enum_vector[input_main_index_array]
	return output_main_enum_maybe_array


def main_enum_to_main_index(input_main_enum_maybe_array):
	input_main_enum_array = np.array(input_main_enum_maybe_array, dtype=object)
	assert np.all((input_main_enum_array >= 0), axis=None, keepdims=False)
	assert np.all((input_main_enum_array < num_main_classes()), axis=None, keepdims=False)

	if (input_main_enum_array.ndim == 0):
		output_main_index_maybe_array = np.int32(input_main_enum_maybe_array)
	else:
		output_main_index_maybe_array = input_main_enum_array.astype(np.int32)
	
	return output_main_index_maybe_array


# If the (i, j) entry of this matrix is true, then class j could serve as occluder for class i.
_main_occlusion_matrix = np.array([
	[True, True, False, False], 
	[True, True, False, False], 
	[False, False, True, False], 
	[True, True, False, False]
], dtype=np.bool_)


def get_occluder_main_index_vector(main_enum_or_index):
	main_index = int(main_enum_or_index)
	assert (main_index >= 0)
	assert (main_index < num_main_classes())

	(occluder_main_index_vector, ) = np.nonzero(_main_occlusion_matrix[main_index])
	return occluder_main_index_vector


#####################
#     SHAPENET      #
#####################

_shapenet_synset_to_main_index_dict = {
	"03001627": 0, 
	"04379243": 1, 
	"02958343": 2, 
}
assert (len(_shapenet_synset_to_main_index_dict) == num_main_classes())


_main_index_to_shapenet_synset_vector = np.empty([num_main_classes()], dtype=object)
for (_shapenet_synset, _main_index) in _shapenet_synset_to_main_index_dict.items():
	assert (_main_index_to_shapenet_synset_vector[_main_index] is None)
	_main_index_to_shapenet_synset_vector[_main_index] = _shapenet_synset
assert np.all(np.not_equal(_main_index_to_shapenet_synset_vector, None), axis=None, keepdims=False)


def main_index_to_shapenet_synset(input_main_index_maybe_array):
	input_main_index_array = np.array(input_main_index_maybe_array, dtype=np.int32)
	assert np.all((input_main_index_array >= 0), axis=None, keepdims=False)
	assert np.all((input_main_index_array < num_main_classes()), axis=None, keepdims=False)

	return _main_index_to_shapenet_synset_vector[input_main_index_array]


def shapenet_synset_to_main_index(input_shapenet_synset_maybe_array):
	input_shapenet_synset_array = np.array(input_shapenet_synset_maybe_array, dtype=object)

	if (input_shapenet_synset_array.ndim == 0):
		output_main_index_maybe_array = np.int32(_shapenet_synset_to_main_index_dict[input_shapenet_synset_maybe_array])
	else:
		output_main_index_maybe_array = (-1) * np.ones(input_shapenet_synset_array.shape, dtype=np.int32)
		for (shapenet_synset, main_index) in _shapenet_synset_to_main_index_dict.items():
			np.copyto(
				output_main_index_maybe_array, main_index, casting="same_kind", 
				where=np.equal(input_shapenet_synset_array, shapenet_synset)
			)
		
		# If this assertion fails, some input shapnet synsets are invalid or not handled by this project 
		# (i.e. not in _shapenet_synset_to_main_index_dict).
		assert np.all((output_main_index_maybe_array >= 0), axis=None, keepdims=False)
	
	return output_main_index_maybe_array


#####################
#       COCO        #
#####################

# This is copied from the "categories" annotation in instances_train2014.json
# Why don't I read it on the fly? Since the annotation file is large, reading it on the fly
# will make importing this script incredibly slow (10+ seconds).
_coco_index_to_coco_id_vector = np.array([
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
	22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 
	46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 
	67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
], dtype=np.int32)


def num_coco_classes():
	return (_coco_index_to_coco_id_vector.size)


_coco_index_to_main_index_dict = {
	2: 2,   # Coco "car" is mapped to main class "CAR"
	7: 2,   # Coco "truck" is mapped to main class "CAR"
	56: 0,  # Coco "chair" is mapped to main class "CHAIR"
	60: 1,  # Coco "dining table" is mapped to main class "TABLE"
	61: 0   # Coco "toilet" is mapped to main class "CHAIR"
}


# Construct the inverse of the vector above, filling -1 at empty spots.
_coco_id_to_coco_index_vector = (-1) * np.ones(
	(np.amax(_coco_index_to_coco_id_vector, axis=None, keepdims=False) + 1), dtype=np.int32
)
for _coco_class_index in range(len(_coco_index_to_coco_id_vector)):
	_coco_class_id = _coco_index_to_coco_id_vector[_coco_class_index]
	_coco_id_to_coco_index_vector[_coco_class_id] = _coco_class_index
del _coco_class_index, _coco_class_id


def coco_id_to_coco_index(input_coco_id_maybe_array):
	input_coco_id_array = np.array(input_coco_id_maybe_array, dtype=np.int32)
	output_coco_index_maybe_array = _coco_id_to_coco_index_vector[input_coco_id_array]
	assert np.all(output_coco_index_maybe_array >= 0, axis=None, keepdims=False)
	return output_coco_index_maybe_array


def coco_index_to_coco_id(input_coco_index_maybe_array):
	input_coco_index_array = np.array(input_coco_index_maybe_array, dtype=np.int32)
	output_coco_id_maybe_array = _coco_index_to_coco_id_vector[input_coco_index_array]
	return output_coco_id_maybe_array


def is_coco_index_handled(input_coco_index_maybe_array):
	input_coco_index_array = np.array(input_coco_index_maybe_array, dtype=np.int32)

	if (input_coco_index_array.ndim == 0):
		output_is_handled_maybe_mask = np.bool_(input_coco_index_maybe_array in _coco_index_to_main_index_dict)
	else:
		output_is_handled_maybe_mask = np.zeros(input_coco_index_array.shape, dtype=np.bool_)
		for (coco_index, main_index) in _coco_index_to_main_index_dict.items():
			np.copyto(
				output_is_handled_maybe_mask, True, casting="same_kind", 
				where=np.equal(input_coco_index_maybe_array, coco_index)
			)
	
	return output_is_handled_maybe_mask


def coco_index_to_main_index(input_coco_index_maybe_array):
	input_coco_index_array = np.array(input_coco_index_maybe_array, dtype=np.int32)

	if (input_coco_index_array.ndim == 0):
		output_main_index_maybe_array = np.int32(_coco_index_to_main_index_dict[input_coco_index_maybe_array])
	else:
		output_main_index_maybe_array = (-1) * np.ones(input_coco_index_array.shape, dtype=np.int32)
		for (coco_index, main_index) in _coco_index_to_main_index_dict.items():
			np.copyto(
				output_main_index_maybe_array, main_index, casting="same_kind", 
				where=np.equal(input_coco_index_array, coco_index)
			)
		
		# If this assertion fails, some input coco indices are invalid or not handled by this project 
		# (i.e. not in _coco_index_to_main_index_dict).
		assert np.all((output_main_index_maybe_array >= 0), axis=None, keepdims=False)
	
	return output_main_index_maybe_array


#####################
#       KITTI       #
#####################

@enum.unique
class KittiClass(enum.IntEnum):
	CAR = 0
	VAN = 1
	TRUCK = 2
	PEDESTRIAN = 3
	SITTER = 4
	CYCLIST = 5
	TRAM = 6
	MISC = 7


def num_kitti_classes():
	return len(KittiClass)


_kitti_index_to_enum_vector = np.empty([num_kitti_classes()], dtype=object)
for _kitti_index in range(num_kitti_classes()):
	_kitti_index_to_enum_vector[_kitti_index] = KittiClass(_kitti_index)
del _kitti_index


def kitti_index_to_kitti_enum(input_kitti_index_maybe_array):
	input_kitti_index_array = np.array(input_kitti_index_maybe_array, dtype=np.int32)
	assert np.all((input_kitti_index_array >= 0), axis=None, keepdims=False)
	assert np.all((input_kitti_index_array < num_kitti_classes()), axis=None, keepdims=False)

	output_kitti_enum_maybe_array = _kitti_index_to_enum_vector[input_kitti_index_array]
	return output_kitti_enum_maybe_array


def kitti_enum_to_kitti_index(input_kitti_enum_maybe_array):
	input_kitti_enum_array = np.array(input_kitti_enum_maybe_array, dtype=object)
	assert np.all((input_kitti_enum_array >= 0), axis=None, keepdims=False)
	assert np.all((input_kitti_enum_array < num_kitti_classes()), axis=None, keepdims=False)

	if (input_kitti_enum_array.ndim == 0):
		output_kitti_index_maybe_array = np.int32(input_kitti_enum_maybe_array)
	else:
		output_kitti_index_maybe_array = input_kitti_enum_array.astype(np.int32)
	
	return output_kitti_index_maybe_array


_kitti_index_to_main_index_dict = {
	0: 2,   # KITTI "CAR" is mapped to main class "CAR"
	1: 2,   # KITTI "VAN" is mapped to main class "CAR"
	2: 2,   # KITTI "TRUCK" is mapped to main class "CAR"
}


def is_kitti_index_handled(input_kitti_index_maybe_array):
	input_kitti_index_array = np.array(input_kitti_index_maybe_array, dtype=np.int32)

	if (input_kitti_index_array.ndim == 0):
		output_is_handled_maybe_mask = np.bool_(input_kitti_index_maybe_array in _kitti_index_to_main_index_dict)
	else:
		output_is_handled_maybe_mask = np.zeros(input_kitti_index_array.shape, dtype=np.bool_)
		for (kitti_index, main_index) in _kitti_index_to_main_index_dict.items():
			np.copyto(
				output_is_handled_maybe_mask, True, casting="same_kind", 
				where=np.equal(input_kitti_index_maybe_array, kitti_index)
			)
	
	return output_is_handled_maybe_mask


def kitti_index_to_main_index(input_kitti_index_maybe_array):
	input_kitti_index_array = np.array(input_kitti_index_maybe_array, dtype=np.int32)

	if (input_kitti_index_array.ndim == 0):
		output_main_index_maybe_array = np.int32(_kitti_index_to_main_index_dict[input_kitti_index_maybe_array])
	else:
		output_main_index_maybe_array = (-1) * np.ones(input_kitti_index_array.shape, dtype=np.int32)
		for (kitti_index, main_index) in _kitti_index_to_main_index_dict.items():
			np.copyto(
				output_main_index_maybe_array, main_index, casting="same_kind", 
				where=np.equal(input_kitti_index_array, kitti_index)
			)
		
		# If this assertion fails, some input kitti indices are invalid or not handled by this project 
		# (i.e. not in _kitti_index_to_main_index_dict).
		assert np.all((output_main_index_maybe_array >= 0), axis=None, keepdims=False)
	
	return output_main_index_maybe_array