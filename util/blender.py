from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bpy

import os
import sys
import numpy as np

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, "../.."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import util.transform


def clear_textures():
	for texture in bpy.data.textures:
		bpy.data.textures.remove(texture, do_unlink=True)


def clear_materials():
	for material in bpy.data.materials:
		bpy.data.materials.remove(material, do_unlink=True)


def clear_objects_except_camera():
	for blender_object in bpy.data.objects:
		if (blender_object.type != "CAMERA"):
			bpy.data.objects.remove(blender_object, do_unlink=True)


def clear_lamps():
	for blender_object in bpy.data.objects:
		if (blender_object.type == "LAMP"):
			blender_object.select = True
	
	for blender_object in bpy.data.objects:
		if (blender_object.type != "LAMP"):
			blender_object.select = False
	
	bpy.ops.object.delete()


def get_default_scene():
	assert (bpy.context.scene != None)
	return bpy.context.scene


def get_default_world():
	default_scene = get_default_scene()
	assert (default_scene.world != None)
	return default_scene.world


def get_default_camera():
	default_scene = get_default_scene()
	assert (default_scene.camera != None)
	return default_scene.camera


def set_environment_lighting_conditions(energy):
	energy = float(energy)
	assert (energy >= 0.0)

	default_world = get_default_world()
	default_world.light_settings.use_environment_light = True
	default_world.light_settings.environment_energy = energy
	default_world.light_settings.environment_color = "PLAIN"


def add_single_lamp(name, energy, location):
	assert isinstance(name, str)
	assert (len(name) > 0)
	energy = float(energy)
	assert (energy >= 0.0)
	assert isinstance(location, tuple)
	assert (len(location) == 3)
	location = (float(location[0]), float(location[1]), float(location[2]))

	lamp_data = bpy.data.lamps.new(name=name, type="POINT")
	lamp_data.energy = energy

	lamp_object = bpy.data.objects.new(name=name, object_data=lamp_data)
	lamp_object.location = location

	default_scene = get_default_scene()
	default_scene.objects.link(lamp_object)


def set_origin_pointing_camera_pose(camera_azimuth_rad, camera_elevation_rad, camera_axial_rotation_rad, camera_distance):
	default_camera = get_default_camera()

	# Rotation and location in Blender means "camera to world" rotation matrix and translation vector
	# However, there is a little tweak. The camera in blender points to -Z direction, while functions in util.transform assume that the 
	# camera points to the +Z direction. So there is a constant difference between the two (which is a rotation around x for 180 degrees)
	camera_to_world_rotation_matrix = util.transform.viewpoint_angles_to_rotation_matrix(camera_azimuth_rad, camera_elevation_rad, camera_axial_rotation_rad)
	x_180_rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
	blender_camera_rotation_matrix = np.dot(camera_to_world_rotation_matrix, x_180_rotation_matrix)
	blender_camera_rotation_quaternion = util.transform.rotation_matrix_to_quaternion(blender_camera_rotation_matrix)

	blender_camera_location_vector = camera_distance * np.reshape(
		np.dot(blender_camera_rotation_matrix, np.array([[0.0], [0.0], [1.0]], dtype=np.float64)), (3, )
	)

	default_camera.location[0] = blender_camera_location_vector[0]
	default_camera.location[1] = blender_camera_location_vector[1]
	default_camera.location[2] = blender_camera_location_vector[2]
	default_camera.rotation_mode = 'QUATERNION'
	default_camera.rotation_quaternion[0] = blender_camera_rotation_quaternion[0]
	default_camera.rotation_quaternion[1] = blender_camera_rotation_quaternion[1]
	default_camera.rotation_quaternion[2] = blender_camera_rotation_quaternion[2]
	default_camera.rotation_quaternion[3] = blender_camera_rotation_quaternion[3]