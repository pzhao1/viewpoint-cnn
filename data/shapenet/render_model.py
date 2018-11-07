"""
Render images for a ShapeNet model.

References:
	Joint embeddings of shapes and images via CNN image purification, Y. Li, H. Su, C. Qi et al.,  2015
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bpy

import os
import sys
import argparse
import numpy as np

# Need to install scikit-image for blender's bundled python.
# First, download get-pip.py from here: https://pip.pypa.io/en/stable/installing/
# Second, <blender_path>/2.xx/python/bin/pythonx.xm get-pip.py
# Third, <blender_path>/2.xx/python/bin/pythonx.xm -m pip install scikit-image
import skimage.io
import skimage.transform

this_file_directory= os.path.dirname(os.path.abspath(__file__))
if (this_file_directory not in sys.path):
	sys.path.append(this_file_directory)

project_base_directory = os.path.abspath(os.path.join(this_file_directory, "../.."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import util.blender
import util.geometry
import util.image
import util.statistics
import data.shapenet.meta_pb2 # If this is not found, bash ../../build_protobuf.sh
import render_config


def construct_argument_parser():
	argument_parser = argparse.ArgumentParser(
		prog="blender", 
		description="Render images for a ShapeNet model."
	)

	# Add arguments to consume blender arguments. Won't be used in this script.
	argument_parser.add_argument(
		"-b", "--background", 
		action="store_true", 
		help="The ShapeNet synset whose images you want to render"
	)

	argument_parser.add_argument(
		"-P", "--python", 
		required=True, 
		help="The Python script to run for blender"
	)

	# Arguments used in this script
	argument_parser.add_argument(
		"model_shapenet_id", 
		help="The ShapeNet ID of the model to render images for"
	)
	argument_parser.add_argument(
		"model_shapenet_path", 
		help="The path of the ShapeNet model to render images for"
	)
	argument_parser.add_argument(
		"model_output_dir", 
		help="The directory where the images should be written"
	)
	argument_parser.add_argument(
		"synset_meta_file_path", 
		help="The path of the file containing protobuf-encoded RSSynsetMeta"
	)
	argument_parser.add_argument(
		"synset_meta_header_length", type=int, 
		help="The encoded length of RSSynsetMetaHeader in the meta file"
	)

	return argument_parser


def build_material_diffuse_color_dictionary():
	material_diffuse_color_dict = {}

	for (material_key, material) in bpy.data.materials.items():
		material_diffuse_color_dict[material_key] = (material.diffuse_color[0], material.diffuse_color[1], material.diffuse_color[2])
	
	return material_diffuse_color_dict


def randomly_add_noise_to_color(original_color, noise_magnitude):
	assert isinstance(original_color, tuple)
	assert (len(original_color) == 3)
	noise_magnitude = float(noise_magnitude)
	assert (noise_magnitude > 0.0)

	color_noise = np.random.uniform(-1.0 * noise_magnitude, noise_magnitude, size=3)
	modified_color_array = np.maximum(np.minimum(np.array(original_color, dtype=np.float64) + color_noise, 1.0), 0.0)
	modified_color = tuple(modified_color_array)

	return modified_color


def randomly_set_material_properties(material_diffuse_color_dict):
	assert isinstance(material_diffuse_color_dict, dict)

	for (material_key, material) in bpy.data.materials.items():
		material_ambient_ratio = np.random.uniform(render_config.MATERIAL_AMBIENT_MIN, render_config.MATERIAL_AMBIENT_MAX)
		material.ambient = material_ambient_ratio

		material_original_diffuse_color = material_diffuse_color_dict[material_key]
		material_modified_diffuse_color = randomly_add_noise_to_color(material_original_diffuse_color, render_config.MATERIAL_COLOR_NOISE)
		# material_modified_diffuse_color = tuple(np.random.uniform(0.0, 1.0, size=3))
		# material_modified_diffuse_color = material_original_diffuse_color
		material.diffuse_color = material_modified_diffuse_color


def randomly_set_lighting_conditions():
	environment_light_energy = np.random.uniform(render_config.ENV_ENERGY_MIN, render_config.ENV_ENERGY_MAX, size=None)
	util.blender.set_environment_lighting_conditions(environment_light_energy)

	util.blender.clear_lamps()

	num_lamps = np.random.randint(low=render_config.NUM_LAMPS_MIN, high=(render_config.NUM_LAMPS_MAX + 1), size=None, dtype=np.int32)
	for lamp_index in range(num_lamps):
		lamp_energy = np.random.uniform(render_config.LAMP_ENERGY_MIN, render_config.LAMP_ENERGY_MAX, size=None)
		lamp_azimuth_rad = np.random.uniform(render_config.LAMP_AZIMUTH_MIN, render_config.LAMP_AZIMUTH_MAX, size=None)
		lamp_elevation_rad  = np.random.uniform(render_config.LAMP_ELEVATION_MIN, render_config.LAMP_ELEVATION_MAX, size=None)
		lamp_distance = np.random.uniform(render_config.LAMP_DISTANCE_MIN, render_config.LAMP_DISTANCE_MAX, size=None)

		lamp_x = (lamp_distance * np.cos(lamp_elevation_rad) * np.cos(lamp_azimuth_rad))
		lamp_y = (lamp_distance * np.cos(lamp_elevation_rad) * np.sin(lamp_azimuth_rad))
		lamp_z = (lamp_distance * np.sin(lamp_elevation_rad))

		util.blender.add_single_lamp("Lamp" + str(lamp_index), lamp_energy, (lamp_x, lamp_y, lamp_z))


def randomly_generate_camera_pose():
	camera_azimuth_rad = np.random.uniform(render_config.CAM_AZIMUTH_MIN, render_config.CAM_AZIMUTH_MAX, size=None)
	camera_elevation_rad = util.statistics.sample_truncated_normal(
		render_config.CAM_ELEVATION_MIN, render_config.CAM_ELEVATION_MAX, 
		render_config.CAM_ELEVATION_MEAN, render_config.CAM_ELEVATION_SIGMA, None
	)
	camera_axial_rotation_rad = 0.0
	camera_distance = np.random.uniform(render_config.CAM_DISTANCE_MIN, render_config.CAM_DISTANCE_MAX, size=None)

	return camera_azimuth_rad, camera_elevation_rad, camera_axial_rotation_rad, camera_distance


def save_rendered_image(output_rgba_image_path, render_rgba_image_array):
	output_rgba_image_path = str(output_rgba_image_path)
	assert (len(output_rgba_image_path) > 4)
	assert (output_rgba_image_path.endswith(".png"))

	assert isinstance(render_rgba_image_array, np.ndarray)
	assert np.issubdtype(render_rgba_image_array.dtype, np.integer)
	(image_size_y, image_size_x, image_num_channels) = render_rgba_image_array.shape
	assert (image_num_channels == 4)

	crop_rectangle = util.geometry.nonzero_bounding_rectangle(render_rgba_image_array[:, :, 3])
	if ((int(crop_rectangle.min_x) <= 0) or (int(crop_rectangle.max_x) >= (render_config.CAM_RESOLUTION_X - 1)) or 
			(int(crop_rectangle.min_y) <= 0) or (int(crop_rectangle.max_y) >= (render_config.CAM_RESOLUTION_Y - 1))):
		# If projection goes all the way to the boundary, then camera is probably too close so that a portion of the object is not captured.
		# Do not accept this image.
		return False
	crop_index_object = util.image.rectangle_to_image_index(crop_rectangle)
	cropped_rgba_image_array = render_rgba_image_array[crop_index_object]

	square_size = int(np.ceil(crop_rectangle.max_size))
	if (square_size < render_config.CROP_SIZE_THRESHOLD):
		# Camera is too far away, and image is too blurred. Do not accept.
		return False
	square_rgba_image_array = np.zeros((square_size, square_size, 4), dtype=np.uint8)
	embedded_rectangle = crop_rectangle.copy()
	embedded_rectangle.apply_translation(
		((float(square_size) / 2.0) - crop_rectangle.center_x), ((float(square_size) / 2.0) - crop_rectangle.center_y)
	)
	embedded_index_object = util.image.rectangle_to_image_index(embedded_rectangle)
	square_rgba_image_array[embedded_index_object] = cropped_rgba_image_array
	
	output_rgba_image_array = skimage.transform.resize(
		square_rgba_image_array, (render_config.IMAGE_SIZE, render_config.IMAGE_SIZE), 
		order=1, preserve_range=True, anti_aliasing=True
	).astype(np.uint8)

	skimage.io.imsave(output_rgba_image_path, output_rgba_image_array)

	return True


def main():
	# Parse command line arguments
	argument_parser = construct_argument_parser()
	argument_namespace = argument_parser.parse_args()

	model_shapenet_id = argument_namespace.model_shapenet_id
	model_shapenet_path = argument_namespace.model_shapenet_path
	model_output_dir = argument_namespace.model_output_dir
	synset_meta_file_path = argument_namespace.synset_meta_file_path
	synset_meta_header_length = argument_namespace.synset_meta_header_length

	assert os.path.isfile(model_shapenet_path)
	os.makedirs(model_output_dir)
	assert os.path.isfile(synset_meta_file_path)
	assert (synset_meta_header_length > 0)

	# Read header portion of the meta file and parse into RSSynsetMetaHeader object.
	synset_meta_prefix_pbobj = data.shapenet.meta_pb2.RSSynsetMeta()
	synset_meta_file = open(synset_meta_file_path, mode="rb")
	synset_meta_prefix_pbobj.ParseFromString(synset_meta_file.read(synset_meta_header_length))
	synset_meta_file.close()
	synset_meta_header_pbobj = synset_meta_prefix_pbobj.header

	# Set render settings.
	default_scene = util.blender.get_default_scene()
	default_scene.render.image_settings.file_format = "PNG"
	default_scene.render.image_settings.color_mode = "RGBA"
	default_scene.render.alpha_mode = "TRANSPARENT"
	default_scene.render.resolution_x = render_config.CAM_RESOLUTION_X
	default_scene.render.resolution_y = render_config.CAM_RESOLUTION_Y
	default_scene.render.resolution_percentage = 100
	# Blender Python API does not allow directly accessing rendered results. Have to save to a temporary file and then read it (I hate this solution!)
	temporary_image_path = os.path.join(this_file_directory, "temp_" + synset_meta_header_pbobj.shapenet_synset + ".png")
	default_scene.render.filepath = temporary_image_path

	# Clear starter objects and lamps.
	util.blender.clear_textures()
	util.blender.clear_materials()
	util.blender.clear_objects_except_camera()

	# Import model.
	if model_shapenet_path.endswith(".obj"):
		bpy.ops.import_scene.obj(filepath=model_shapenet_path, axis_forward="-Z", axis_up="Y")
	
	elif model_shapenet_path.endswith(".ply"):
		bpy.ops.import_mesh.ply(filepath=model_shapenet_path)
		# .ply files do not have materials. They have vertex colors instead.
		# Create a temporary material.
		dummy_material = bpy.data.materials.new(name="dummy_material")
		dummy_material.use_transparency = False
		dummy_material.use_raytrace = True
		dummy_material.use_mist = False
		# dummy_material.diffuse_color = (0.3, 0.3, 0.3)
		dummy_material.diffuse_intensity = 1.0
		dummy_material.diffuse_shader = 'LAMBERT'
		dummy_material.specular_color = (1.0, 1.0, 1.0)
		dummy_material.specular_intensity = 0.1
		dummy_material.specular_hardness = 8
		dummy_material.specular_shader = 'PHONG'
		dummy_material.ambient = 1.0
		dummy_material.emit = 0.0
		dummy_material.translucency = 0.0

		# Let the material display vertex color.
		dummy_material.use_vertex_color_paint = True

		# Associate this material to all mesh objects.
		for blender_object in bpy.data.objects:
			if (blender_object.type == "MESH"):
				blender_object.data.materials.append(dummy_material)
	
	else:
		raise ValueError("Cannot handle model type: %s" %(model_shapenet_path, ))
	
	material_diffuse_color_dict = build_material_diffuse_color_dictionary()

	# Construct an RSSynsetMeta object with an empty header and only one RSModelMeta object in model_meta_list.
	# According to the encoding rules of protobuf, the serialized string of this RSSynsetMeta object will be exactly the same 
	# as the serialized string of a single RSModelMeta item in an RSSynsetMeta object with multiple items in model_meta_list.
	# Therefore, in order to add the RSModelMeta of this model to the accumulating RSSynsetMeta, we can simply append 
	# the serlized string of this "suffix" RSSynsetMeta to synset_meta_file.
	# https://developers.google.com/protocol-buffers/docs/encoding
	synset_meta_suffix_pbobj = data.shapenet.meta_pb2.RSSynsetMeta()
	model_meta_pbobj = synset_meta_suffix_pbobj.model_meta_list.add()
	model_meta_pbobj.shapenet_id = model_shapenet_id

	# Main loop
	accepted_config_index = 0
	for config_index in range(synset_meta_header_pbobj.num_configs):
		print("\nRendering image %d out of %d total\n" % (config_index, synset_meta_header_pbobj.num_configs))

		randomly_set_material_properties(material_diffuse_color_dict)
		randomly_set_lighting_conditions()

		(camera_azimuth_rad, camera_elevation_rad, camera_axial_rotation_rad, camera_distance) = randomly_generate_camera_pose()
		assert ((camera_azimuth_rad >= 0.0) and (camera_azimuth_rad <= (2.0 * np.pi)))
		assert ((camera_elevation_rad >= (- np.pi / 2.0)) and (camera_elevation_rad <= (np.pi / 2.0)))
		assert ((camera_axial_rotation_rad >= 0.0) and (camera_axial_rotation_rad <= (2.0 * np.pi)))
		assert (camera_distance >= 0.0)

		util.blender.set_origin_pointing_camera_pose(camera_azimuth_rad, camera_elevation_rad, camera_axial_rotation_rad, camera_distance)
		
		bpy.ops.render.render(write_still=True)
		render_rgba_image_array = skimage.io.imread(temporary_image_path)
		output_rgba_image_path = os.path.join(model_output_dir, ("%d.png" % (config_index, )))

		is_image_accepted = save_rendered_image(output_rgba_image_path, render_rgba_image_array)
		if (not is_image_accepted):
			continue
		else:
			accepted_config_index = accepted_config_index + 1
		
		record_meta_pbobj = model_meta_pbobj.record_meta_list.add()
		record_meta_pbobj.config_index = accepted_config_index
		record_meta_pbobj.camera_pose.azimuth_rad = camera_azimuth_rad
		record_meta_pbobj.camera_pose.elevation_rad = camera_elevation_rad
		record_meta_pbobj.camera_pose.axial_rotation_rad = camera_axial_rotation_rad
		record_meta_pbobj.camera_pose.distance = camera_distance
		record_meta_pbobj.image_relative_path = os.path.relpath(output_rgba_image_path, os.path.dirname(synset_meta_file_path))
	
	synset_meta_file = open(synset_meta_file_path, mode="ab")
	synset_meta_file.write(synset_meta_suffix_pbobj.SerializeToString())
	synset_meta_file.close()


if __name__=="__main__":
	main()


# Legacy code for visualizing camera pose distributions

# def _visualize_camera_pose_distribution():

# 	camera_azimuth_rad = np.random.uniform(render_config.CAM_AZIMUTH_MIN, render_config.CAM_AZIMUTH_MAX, size=300000)
# 	camera_elevation_rad = util.statistics.sample_truncated_normal(
# 		render_config.CAM_ELEVATION_MIN, render_config.CAM_ELEVATION_MAX, 
# 		render_config.CAM_ELEVATION_MEAN, render_config.CAM_ELEVATION_SIGMA, 300000
# 	)
# 	camera_distance = np.random.uniform(render_config.CAM_DISTANCE_MIN, render_config.CAM_DISTANCE_MAX, size=300000)

# 	import matplotlib.pyplot
	
# 	assert (not np.any(camera_azimuth_rad < render_config.CAM_AZIMUTH_MIN, axis=None))
# 	assert (not np.any(camera_azimuth_rad > render_config.CAM_AZIMUTH_MAX, axis=None))
# 	matplotlib.pyplot.hist(camera_azimuth_rad, 50)
# 	matplotlib.pyplot.title("Azimuth")
# 	matplotlib.pyplot.show()
	
# 	assert (not np.any(camera_elevation_rad < render_config.CAM_ELEVATION_MIN, axis=None))
# 	assert (not np.any(camera_elevation_rad > render_config.CAM_ELEVATION_MAX, axis=None))
# 	matplotlib.pyplot.hist(camera_elevation_rad, 50)
# 	matplotlib.pyplot.title("Elevation")
# 	matplotlib.pyplot.show()

# 	assert (not np.any(camera_distance < render_config.CAM_DISTANCE_MIN, axis=None))
# 	assert (not np.any(camera_distance > render_config.CAM_DISTANCE_MAX, axis=None))
# 	matplotlib.pyplot.hist(camera_distance, 50)
# 	matplotlib.pyplot.title("Distance")
# 	matplotlib.pyplot.show()