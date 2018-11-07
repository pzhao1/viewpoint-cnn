from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import trimesh
import numpy as np

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, "../.."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import util.geometry
import util.mesh


def generate_icosahedron_mesh_test1():
	icosahedron_mesh = util.mesh.generate_icosahedron_mesh()
	util.mesh.visualize_mesh(icosahedron_mesh)


def generate_subdivided_icosahedron_mesh_test1():
	triangulated_sphere_mesh = util.mesh.generate_subdivided_icosahedron_mesh(4)
	util.mesh.visualize_mesh(triangulated_sphere_mesh)


def compute_generalized_moments_test1():
	# test_mesh_shapenet_id = "4ce0cbd82a8f86a71dffa0a43719d0b5"  # 1-fold discrete z-rotation symmetry (or no symmetry)
	# test_mesh_shapenet_id = "f9f9d2fda27c310b266b42a2f1bdd7cf"  # 2-fold discrete z-rotation symmetry
	# test_mesh_shapenet_id = "8010b1ce4e4b1a472a82acb89c31cb53"  # 3-fold discrete z-rotation symmetry
	# test_mesh_shapenet_id = "2ee72f0fa8848523f1d2a696b973c343"  # 4-fold discrete z-rotation symmetry
	# test_mesh_shapenet_id = "89aa38d569b025b2dd70fcdaf3665b80"  # 5-fold discrete z-rotation symmetry
	test_mesh_shapenet_id = "4d3bdfe96a1d334d3c329e0c5f819d20"  # 8-fold discrete z-rotation symmetry
	# test_mesh_shapenet_id = "3ada04a73dd3fe98c520ac3fa0a4f674"  # Continuous z-rotation symmetry

	test_mesh = trimesh.load_mesh(os.path.join("example_mesh", test_mesh_shapenet_id, "models", "model_normalized.obj"))

	# ShapeNet models use the "Y up, -Z forward" scheme, but my utilitiy functions use the "Z up, Y forward" scheme.
	# Rotate around x by 90 degrees to compensate this difference.
	x_90_transform_matrix = np.array([
		[1.0, 0.0, 0.0, 0.0], 
		[0.0, 0.0, -1.0, 0.0], 
		[0.0, 1.0, 0.0, 0.0], 
		[0.0, 0.0, 0.0, 1.0]
	], dtype=np.float64)
	test_mesh.apply_transform(x_90_transform_matrix)
	util.mesh.visualize_mesh(test_mesh)

	# Compute generalized moments
	max_moment_degree = 8
	moment_sh_mixture_dict = util.mesh.compute_generalized_moments(test_mesh, max_moment_degree)

	# Visualize moment function
	visualize_moment_degree = 8
	visualize_moment_sh_mixture = moment_sh_mixture_dict[visualize_moment_degree]

	sample_direction_mesh = util.mesh.generate_subdivided_icosahedron_mesh(4)
	sample_azimuth_vector = util.geometry.compute_points_azimuth(sample_direction_mesh.vertices)
	sample_colatitude_vector = util.geometry.compute_points_colatitude(sample_direction_mesh.vertices)
	test_moment_value_vector = 100*visualize_moment_sh_mixture.evaluate(sample_azimuth_vector, sample_colatitude_vector)
	util.mesh.visualize_spherical_function(sample_direction_mesh, test_moment_value_vector)


def detect_z_rotation_symmetry_test1():
	# test_mesh_shapenet_id = "4ce0cbd82a8f86a71dffa0a43719d0b5"  # 1-fold discrete z-rotation symmetry (or no symmetry)
	# test_mesh_shapenet_id = "f9f9d2fda27c310b266b42a2f1bdd7cf"  # 2-fold discrete z-rotation symmetry
	# test_mesh_shapenet_id = "8010b1ce4e4b1a472a82acb89c31cb53"  # 3-fold discrete z-rotation symmetry.
	# test_mesh_shapenet_id = "2ee72f0fa8848523f1d2a696b973c343"  # 4-fold discrete z-rotation symmetry
	# test_mesh_shapenet_id = "89aa38d569b025b2dd70fcdaf3665b80"  # 5-fold discrete z-rotation symmetry
	# test_mesh_shapenet_id = "4d3bdfe96a1d334d3c329e0c5f819d20"  # 8-fold discrete z-rotation symmetry
	test_mesh_shapenet_id = "3ada04a73dd3fe98c520ac3fa0a4f674"  # Continuous z-rotation symmetry

	test_mesh = trimesh.load_mesh(os.path.join("example_mesh", test_mesh_shapenet_id, "models", "model_normalized.obj"))
	print("Test mesh has %d vertices and %d faces" % (test_mesh.vertices.shape[0], test_mesh.faces.shape[0]))

	# ShapeNet models use the "Y up, -Z forward" scheme, but my utilitiy functions use the "Z up, Y forward" scheme.
	# Rotate around x by 90 degrees to compensate this difference.
	x_90_transform_matrix = np.array([
		[1.0, 0.0, 0.0, 0.0], 
		[0.0, 0.0, -1.0, 0.0], 
		[0.0, 1.0, 0.0, 0.0], 
		[0.0, 0.0, 0.0, 1.0]
	], dtype=np.float64)
	test_mesh.apply_transform(x_90_transform_matrix)
	util.mesh.visualize_mesh(test_mesh)

	max_degree = 8
	detect_rel_tolerance = 0.4
	verify_num_sample_points = 10000
	verify_rel_tolerance = 0.05
	(has_continuous_symmetry, discrete_symmetry_fold) = util.mesh.detect_z_rotation_symmetry(
		test_mesh, max_degree, detect_rel_tolerance, verify_num_sample_points, verify_rel_tolerance
	)
	print("has_continuous_symmetry: ", has_continuous_symmetry)
	print("discrete_symmetry_fold: ", discrete_symmetry_fold)