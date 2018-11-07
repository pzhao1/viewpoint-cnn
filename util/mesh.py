from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import numpy as np
import scipy.spatial
import scipy.special
import trimesh
import matplotlib.pyplot
import mpl_toolkits.mplot3d

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, ".."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import util.geometry
import util.integrate
import util.sph_harm
import util.transform


def visualize_mesh(mesh):
	assert isinstance(mesh, trimesh.base.Trimesh)

	figure = matplotlib.pyplot.figure()
	axes = figure.add_subplot(1, 1, 1, projection="3d")

	# axes.plot_trisurf(
	# 	mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], 
	# 	triangles=mesh.faces
	# )

	triangle_vertex_tensor = mesh.vertices[mesh.faces, :]
	collection = mpl_toolkits.mplot3d.art3d.Poly3DCollection(triangle_vertex_tensor, facecolors=(mesh.visual.face_colors / 255.0), edgecolors=[0.0, 0.0, 0.0, 0.2])
	axes.add_collection(collection)

	axes.set_xlabel("X")
	axes.set_ylabel("Y")
	axes.set_zlabel("Z")

	bounding_box_center = np.array(mesh.bounds.mean(axis=0), dtype=np.float64)
	maximum_extent = np.amax(mesh.extents, axis=None)
	axes.set_xlim(bounding_box_center[0] - (maximum_extent / 2.0), bounding_box_center[0] + (maximum_extent / 2.0))
	axes.set_ylim(bounding_box_center[1] - (maximum_extent / 2.0), bounding_box_center[1] + (maximum_extent / 2.0))
	axes.set_zlim(bounding_box_center[2] - (maximum_extent / 2.0), bounding_box_center[2] + (maximum_extent / 2.0))

	matplotlib.pyplot.show(block=False)


def visualize_spherical_function(sample_direction_mesh, sample_value_vector):
	assert isinstance(sample_direction_mesh, trimesh.base.Trimesh)
	assert isinstance(sample_value_vector, np.ndarray)
	assert np.issubdtype(sample_value_vector.dtype, np.floating)
	assert (sample_value_vector.ndim == 1)
	assert (sample_value_vector.size == sample_direction_mesh.vertices.shape[0])

	sample_value_mesh = sample_direction_mesh.copy()
	sample_value_mesh.vertices = sample_value_mesh.vertices / np.linalg.norm(sample_value_mesh.vertices, ord=2, axis=1, keepdims=True)
	sample_value_mesh.vertices = sample_value_mesh.vertices * np.expand_dims(np.abs(sample_value_vector), 1)

	face_value_vector = np.sum(sample_value_vector[sample_value_mesh.faces], axis=1, dtype=np.float64) / 3.0
	face_color_matrix = np.stack([
		(face_value_vector > 0.0).astype(np.float64), 
		np.zeros((face_value_vector.size, ), dtype=np.float64), 
		(face_value_vector <= 0.0).astype(np.float64)
	], axis=1)
	sample_value_mesh.visual.face_colors = face_color_matrix

	visualize_mesh(sample_value_mesh)


def generate_icosahedron_mesh():
	"""
	Generate a regular icosahedron mesh with edge length 2.
	"""
	vertex_matrix = util.geometry.get_icosahedron_vertices()
	convex_hull = scipy.spatial.ConvexHull(vertex_matrix)
	icosahedron_mesh = trimesh.base.Trimesh(vertices=vertex_matrix, faces=convex_hull.simplices)
	return icosahedron_mesh


def generate_subdivided_icosahedron_mesh(num_divide_times):
	# Check arguments. 
	# The upper limit for num_divide_times is added to restrict exponentially large resource consumptions.
	num_divide_times = int(num_divide_times)
	assert (num_divide_times >= 0)
	assert (num_divide_times <= 8)

	result_mesh = generate_icosahedron_mesh()
	for _ in range(num_divide_times):
		trimesh.remesh.subdivide(result_mesh, face_index=None)
		result_mesh.vertices = result_mesh.vertices / np.linalg.norm(result_mesh.vertices, ord=2, axis=1, keepdims=True)
	
	return result_mesh


def compute_generalized_moments(mesh, max_degree):
	"""
	Given a triangle mesh, efficiently compute its generalized moments up to max_degree.

	Inputs:
		mesh      : A Trimesh object containing a triangulated mesh.
		max_degree: An even integer specifying the maximum degree of generalized moment to compute.
		            Exceptions will be thrown if max_degree is too large (>10 for now), since the triangle quadrature method 
		            needed for surface integration does not support large degrees.
	
	Outputs:
		moment_sh_mixture_dict: A Python dictionary that maps moment degree to a util.sph_harm.SHMixture object for that moment.
	
	References:
		Accurate detection of symmetries in 3D shapes, Martinet et al., 2006
	"""
	assert isinstance(mesh, trimesh.base.Trimesh)
	centered_mesh = mesh.copy()
	centered_mesh.apply_translation(-1.0 * mesh.centroid)

	max_degree = int(max_degree)
	assert (max_degree >= 2)
	assert ((max_degree % 2) == 0)

	# Compute standard quadrature points.
	# The result "quad_std_point_matrix" has dimension (num_quad_points, 2), and "quad_weight_vector" has dimension (num_quad_points, )
	# Notice that if max_degree is too large, there may not be quadrature points for that degree, and an exception will be thrown.
	(quad_std_point_matrix, quad_weight_vector) = util.integrate.triangle_quadrature_standard_points(max_degree * 2)

	# Compute triangles and their areas in the mesh.
	# The result "triangle_vertex_tensor" has dimension (num_triangles, 3, 3), and "triangle_area_vector" has dimension (num_triangles, )
	triangle_vertex_tensor = centered_mesh.vertices[centered_mesh.faces, :]
	triangle_area_vector = util.geometry.compute_triangles_area(triangle_vertex_tensor)

	# Compute quadrature evaluation points for each face.
	# The result "quad_eval_point_tensor" has dimension (num_triangles, num_quad_points, 3)
	quad_eval_point_tensor = util.integrate.triangle_quadrature_evaluation_points(triangle_vertex_tensor, quad_std_point_matrix)
	quad_eval_point_norm_tensor = np.linalg.norm(quad_eval_point_tensor, ord=2, axis=2, keepdims=True)

	# Compute the azimuth and colatitude of each quadrature evaluation point.
	# The results "quad_eval_azimuth_matrix" and "quad_eval_colatitude_matrix" has dimension (num_triangles, num_quad_points)
	quad_eval_azimuth_matrix = util.geometry.compute_points_azimuth(quad_eval_point_tensor)
	quad_eval_colatitude_matrix = util.geometry.compute_points_colatitude(quad_eval_point_tensor)

	# For each quadrature point, compute the "D" coefficients in equation (5) of referenced paper, which are zonal spherical harmonic
	# transform factors for the rotations that map the z-axis to each quadrature evaluation point.
	# The result "D_tensor" has dimension (num_triangles, num_quad_points, num_sh_components)
	D_tensor = util.sph_harm.sh_zonal_rotation_transform(quad_eval_azimuth_matrix, quad_eval_colatitude_matrix, max_degree)

	# Compute the moments 2, 4, ..., max_degree.
	# Using notation in the referenced paper, except:
	#     "L" is used in place of "l" to destinguish from number 1.
	#     Superscripts and subscripts are both separated by underscore. For example, S^L_p is named S_L_p
	moment_sh_mixture_dict = {}
	for p in range(1, (max_degree // 2) + 1):
		p_float = float(p)
		s_norm_to_2p_tensor = np.power(quad_eval_point_norm_tensor, 2.0 * p_float)

		M_2p_sh_mixture = util.sph_harm.SHMixture()
		for L in range(p + 1):
			L_float = float(L)

			# Compute S^l_p in Equation (3)
			k_vector = np.arange(L, (2 * L) + 1, 1, dtype=np.int32)
			k_float_vector = k_vector.astype(np.float64)
			log_computed_term_vector = np.exp(
				# Use log to compute potentially large terms, such as exponentials and fatorials.
				(2.0 * p_float + 1.0) * np.log(2.0) + 
				scipy.special.gammaln(p_float + 1.0) + 
				scipy.special.gammaln(2.0 * k_float_vector + 1.0) + 
				scipy.special.gammaln(p_float + k_float_vector - L_float + 1.0) - 
				scipy.special.gammaln(2.0 * (p_float + k_float_vector - L_float) + 1.0 + 1.0) - 
				scipy.special.gammaln(k_float_vector - L_float + 1.0) - 
				scipy.special.gammaln(k_float_vector + 1.0) - 
				scipy.special.gammaln(2.0 * L_float - k_float_vector + 1.0) - 
				2.0 * L_float * np.log(2.0)
			)
			S_L_p = np.sqrt((4.0 * L_float + 1.0) * np.pi) * np.sum(
				np.power(-1.0, k_float_vector) * log_computed_term_vector, 
				axis=None, dtype=np.float64, keepdims=False
			)

			# Compute Equation 5 (vectorized for all m)
			(m_begin_linear_index, m_end_linear_index) = util.sph_harm.sh_degree_to_linear_index_range(2 * L)
			C_2p_2L_vector = S_L_p * np.sum(
				(
					np.reshape(triangle_area_vector, (-1, 1, 1)) *
					np.reshape(quad_weight_vector, (1, -1, 1)) * 
					s_norm_to_2p_tensor * 
					D_tensor[:, :, m_begin_linear_index:m_end_linear_index]
				), 
				axis=(0, 1), dtype=np.float64, keepdims=False
			)
			M_2p_sh_mixture.set_degree_weights(2 * L, C_2p_2L_vector)
		
		moment_sh_mixture_dict[2 * p] = M_2p_sh_mixture
	
	return moment_sh_mixture_dict




def verify_z_rotation_symmetry(mesh, rotation_angle_rad, num_sample_points, relative_tolerance):
	"""
	Verify that a mesh has z-rotation symmetry of a given angle.

	Inputs:
		mesh              : The mesh whose symmetries we want to verify.
		rotation_angle_rad: The angle (in radians) of the discrete z-rotation symmetry we want to verify.
		num_sample_points : Number of points to sample when estimating Hausdorff distance. More sample points take longer to run, but give a more 
		                    accurate estimation of Hausdorff distance (and thus a better chance to reject false symmetries). Must be a multiple of 100.
		                    Notice that a (verify_num_sample_points, verify_num_sample_points) matrix will be constructed in the process. 
		                    Lower this number if your computer runs out of memory.
		relative_tolerance: Threshold that accounts for numerical errors and approximate symmetries.
		                    When the Hausdorff distance between original and rotated meshes is <= (relative_tolerance * average_sample_point_norm), 
		                    the input mesh is considered symmetric.
	
	References:
		https://en.wikipedia.org/wiki/Hausdorff_distance
	

	Notes: Here are some statistics from some testing runs with num_sample_points=15,000. 
	They may be helpful for choosing the relative_tolerance parameter.

	test_mesh_shapenet_id                gt_fold    input_fold    (corrected_hausdorff_distance / avg_all_point_norm)
	--------------------------------------------------------------------------------------------------------------------------------------------
	4ce0cbd82a8f86a71dffa0a43719d0b5     1          1             0.00303547115369
	f9f9d2fda27c310b266b42a2f1bdd7cf     2          2             0.00447950502934
	8010b1ce4e4b1a472a82acb89c31cb53     3          3             0.00385930072605
	2ee72f0fa8848523f1d2a696b973c343     4          4             0.0088917345559
	89aa38d569b025b2dd70fcdaf3665b80     5          5             0.00737159002243
	4d3bdfe96a1d334d3c329e0c5f819d20     8          8             0.00465042317493

	3ada04a73dd3fe98c520ac3fa0a4f674     cont.      2             0.00495834583556
	3ada04a73dd3fe98c520ac3fa0a4f674     cont.      3             0.00702951593181
	3ada04a73dd3fe98c520ac3fa0a4f674     cont.      4             0.00416841113398
	3ada04a73dd3fe98c520ac3fa0a4f674     cont.      5             0.00399674336135
	3ada04a73dd3fe98c520ac3fa0a4f674     cont.      6             0.00239513239423
	3ada04a73dd3fe98c520ac3fa0a4f674     cont.      7             0.00498252586002
	3ada04a73dd3fe98c520ac3fa0a4f674     cont.      8             0.00418505703717
	3ada04a73dd3fe98c520ac3fa0a4f674     cont.      9             0.00280460680357

	4ce0cbd82a8f86a71dffa0a43719d0b5     1          2             0.423858513209
	f9f9d2fda27c310b266b42a2f1bdd7cf     2          3             1.04706816021
	8010b1ce4e4b1a472a82acb89c31cb53     3          4             0.465406599977
	2ee72f0fa8848523f1d2a696b973c343     4          5             0.34145570819
	89aa38d569b025b2dd70fcdaf3665b80     5          6             0.16655772361
	4d3bdfe96a1d334d3c329e0c5f819d20     8          9             0.0296247707212

	Notice as the number of folds gets larger and larger it is increasingly difficult to distinguish between
	fold and fold+1.
	"""
	assert isinstance(mesh, trimesh.base.Trimesh)
	centered_mesh = mesh.copy()
	centered_mesh.apply_translation(-1.0 * mesh.centroid)

	rotation_angle_rad = float(rotation_angle_rad)

	relative_tolerance = float(relative_tolerance)
	assert (relative_tolerance >= 0.0)
	assert (relative_tolerance <= 1.0)

	num_sample_points = int(num_sample_points)
	assert (num_sample_points > 0)
	assert ((num_sample_points % 100) == 0)

	z_unit_vector = np.array([0.0, 0.0, 1.0], dtype=np.float64)
	rotation_matrix = util.transform.axis_angle_to_rotation_matrix(z_unit_vector, rotation_angle_rad)
	rigid_transform_matrix = np.eye(4, 4, dtype=np.float64)
	rigid_transform_matrix[0:3, 0:3] = rotation_matrix
	rotated_centered_mesh = centered_mesh.copy()
	rotated_centered_mesh.apply_transform(rigid_transform_matrix)

	# Sample points from original and rotated meshes, and compute closest distance from points of one mesh to the other.
	# Trimesh has a bug in trimesh.proximity.closest_point(), which crashes randomly.
	# Therefore, I am using the shortest distance between to sets of sample points to approximate Hausdorff distance.
	sample_point_matrix = trimesh.sample.sample_surface(centered_mesh, num_sample_points)
	rotated_sample_point_matrix = trimesh.sample.sample_surface(rotated_centered_mesh, num_sample_points)

	point_distance_matrix = util.geometry.compute_points_distance(sample_point_matrix, rotated_sample_point_matrix)
	point_distance0_vector = np.amin(point_distance_matrix, axis=1, keepdims=False)
	point_distance1_vector = np.amin(point_distance_matrix, axis=0, keepdims=False)

	# Estimate Hausdorff distance between original and rotated meshes.
	# Instead of using the maximum distance, I use the average of the maximum 1% of the distances for stability.
	max_1percent_start_index = 99 * (num_sample_points // 100)
	partitioned_point_distance0_vector = np.partition(point_distance0_vector, max_1percent_start_index, axis=None)
	partitioned_point_distance1_vector = np.partition(point_distance1_vector, max_1percent_start_index, axis=None)
	hausdorff_distance = np.maximum(
		np.average(partitioned_point_distance0_vector[max_1percent_start_index:], axis=None), 
		np.average(partitioned_point_distance1_vector[max_1percent_start_index:], axis=None)
	)

	# The approximate above is obviously not accurate, and one way to estimate the error is to perform samples 
	# on the same mesh twice, and compute the "distance" between the two sets, which shouldn't be there without approximation.
	control_sample_point_matrix = trimesh.sample.sample_surface(centered_mesh, num_sample_points)
	control_point_distance_matrix = util.geometry.compute_points_distance(sample_point_matrix, control_sample_point_matrix)
	control_point_distance0_vector = np.amin(control_point_distance_matrix, axis=1, keepdims=False)
	control_point_distance1_vector = np.amin(control_point_distance_matrix, axis=0, keepdims=False)
	control_partitioned_point_distance0_vector = np.partition(control_point_distance0_vector, max_1percent_start_index, axis=None)
	control_partitioned_point_distance1_vector = np.partition(control_point_distance1_vector, max_1percent_start_index, axis=None)
	control_hausdorff_distance = np.maximum(
		np.average(control_partitioned_point_distance0_vector[max_1percent_start_index:], axis=None), 
		np.average(control_partitioned_point_distance1_vector[max_1percent_start_index:], axis=None)
	)

	# Compute decision criterion
	# Notice that the approximation error and the real hausdorff distance are roughly "orthongal".
	# The approximation errors mostly lie on the surface of the mesh, while the real hausdorff errors are perpendicular to
	# the surface. Therefore, use a square difference instead of direct subtraction.
	corrected_hausdorff_distance = np.sqrt(np.maximum(0.0, (np.square(hausdorff_distance) - np.square(control_hausdorff_distance))))
	avg_all_point_norm = np.average(np.linalg.norm(sample_point_matrix, ord=2, axis=1, keepdims=False), axis=None)
	
	return ((corrected_hausdorff_distance / avg_all_point_norm) <= relative_tolerance)


def detect_z_rotation_symmetry(mesh, max_degree, detect_rel_tolerance, verify_num_sample_points, verify_rel_tolerance):
	"""
	Efficiently detect discrete or continuous z-rotation symmetry around the z-axis.

	Inputs:
		mesh      : The mesh whose z-rotation symmetries we want to detect.
		max_degree: The maximum degree of spherical harmonics to use (see reference and implementation).
		            Larger degree can detect finer discrete symmetry, but is also more expensive to run.
		            For example, max_degree = 8 can detect 8-fold discrete symmetry at most. 
		            If the input has 10-fold discrete symmetry, then the algorithm is likely to say the input has continuous symmetry.
		            Exceptions will be thrown if max_degree is too large (>10 for now), since the triangle quadrature method needed for 
		            surface integration does not support large degrees. 
		            Recommended value: 8
		
		detect_rel_tolerance: Relative tolerance used in the detection phase. Will be passed to sphere.SHMixture.detect_z_rotation_symmetry().
		                      Recommended value: 0.4
		verify_num_sample_points: Number of points to sample during verification. Will be passed to verify_z_rotation_symmetry(). Must be a multiple of 100.
		                      Notice that a (verify_num_sample_points, verify_num_sample_points) matrix will be constructed in the process. 
		                      Lower this number if your computer runs out of memory.
		verify_rel_tolerance: Relative tolerance used in the verification phase. Will be paseed to verify_z_rotation_symmetry().
		                      Recommended value: 0.05
	
	Output:
		has_continuous_symmetry : Boolean indicating whether or not the input mesh has continuous z-rotation symmetry.
		discrete_symmetry_fold  : If the mesh does not have continuous z-rotation symmetry, this is the largest fold for its discrete symmetry.
		                          If the mesh does has continuous z-rotation symmetry, this is meaningless and the returned value is not specified.

	References:
		Accurate detection of symmetries in 3D shapes, Martinet et al., 2006
	"""
	assert isinstance(mesh, trimesh.base.Trimesh)

	max_degree = int(max_degree)
	assert (max_degree >= 2)
	assert ((max_degree % 2) == 0)

	detect_rel_tolerance = float(detect_rel_tolerance)
	assert (detect_rel_tolerance >= 0.0)
	assert (detect_rel_tolerance <= 1.0)

	verify_num_sample_points = int(verify_num_sample_points)

	verify_rel_tolerance = float(verify_rel_tolerance)
	assert (verify_rel_tolerance >= 0.0)
	assert (verify_rel_tolerance <= 1.0)	
	
	moment_sh_mixture_dict = compute_generalized_moments(mesh, max_degree)

	may_have_continuous_symmetry = True
	discrete_symmetry_fold_candidate = None
	for (moment_degree, moment_sh_mixture) in moment_sh_mixture_dict.items():
		# In the generlized moment, different degrees of spherical harmonics may have different scales.
		# For example the coefficients for 2st-degree components may be around 1e-2, but the coefficients 
		# for the 6th-degree components may be around 1e-6.
		# Therefore, it is best to rescale the weights in the moment to the same level before detecting symmetry.
		rescaled_moment_sh_mixture = moment_sh_mixture.copy()
		for sh_degree in range(0, moment_degree + 1, 2):
			# Only coefficients for even degrees are non-zero in generalized momenets by construction.
			degree_weight_vector = moment_sh_mixture.get_degree_weights(sh_degree)
			max_abs_degree_weight = np.amax(np.abs(degree_weight_vector), axis=None, keepdims=False)
			rescaled_moment_sh_mixture.set_degree_weights(sh_degree, degree_weight_vector / max_abs_degree_weight)
		
		(moment_is_zonal, moment_discrete_symmetry_fold) = rescaled_moment_sh_mixture.detect_z_rotation_symmetry(np.inf, detect_rel_tolerance)

		may_have_continuous_symmetry = (may_have_continuous_symmetry and moment_is_zonal)

		if (not moment_is_zonal):
			if (discrete_symmetry_fold_candidate is None):
				discrete_symmetry_fold_candidate = moment_discrete_symmetry_fold
			else:
				discrete_symmetry_fold_candidate =  math.gcd(discrete_symmetry_fold_candidate, moment_discrete_symmetry_fold)
	
	# Verify symmetry.
	if (may_have_continuous_symmetry):
		has_continuous_symmetry = (
			# To verify continuous symmetry, verify rotation symmetry for several angles.
			verify_z_rotation_symmetry(mesh, (2.0 * np.pi / 5.0), verify_num_sample_points, verify_rel_tolerance) and
			verify_z_rotation_symmetry(mesh, (4.0 * np.pi / 7.0), verify_num_sample_points, verify_rel_tolerance) and
			verify_z_rotation_symmetry(mesh, (7.0 * np.pi / 11.0), verify_num_sample_points, verify_rel_tolerance) and
			verify_z_rotation_symmetry(mesh, (11.0 * np.pi / 13.0), verify_num_sample_points, verify_rel_tolerance)
		)
		if (has_continuous_symmetry):
			discrete_symmetry_fold = None
		else:
			# It is hard to decide what to do in this case.
			# If generalized moments says there is a continuous geometry, but verification denied it, then it is likely that
			# detect_rel_tolerance is set to high, so some of the non-zonal spherical harmonic components are missed.
			# In this case, the model probably has some discrete symmetry. We just don't know what it is.
			# One way to address this problem is to re-run detection with a lower relative tolerance, but here I am using 
			# the simplest solution - conservatively declare that there is no symmetry at all.
			discrete_symmetry_fold = 1
	
	else:
		has_continuous_symmetry = False
		if (discrete_symmetry_fold_candidate == 1):
			# No need to verify 1-fold discrete symmetry
			discrete_symmetry_fold = 1
		else:
			is_discrete_symmetry_verified = verify_z_rotation_symmetry(
				mesh, (2.0 * np.pi / float(discrete_symmetry_fold_candidate)), verify_num_sample_points, verify_rel_tolerance
			)
			if (is_discrete_symmetry_verified):
				discrete_symmetry_fold = discrete_symmetry_fold_candidate
			else:
				discrete_symmetry_fold = 1
	
	return (has_continuous_symmetry, discrete_symmetry_fold)
	

		