from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.linalg


def is_3d_point(point):
	"""
	Can also be doubled as "is_translation_vector", since they are the same thing.
	"""
	if not isinstance(point, np.ndarray):
		return False
	if not np.issubdtype(point.dtype, np.floating):
		return False
	if (point.shape != (3, )):
		return False
	return True


def is_rotation_matrix(matrix_to_test, tolerance=1e-4):
	tolerance = float(tolerance)
	assert (tolerance > 0.0)

	if (not isinstance(matrix_to_test, np.ndarray)):
		return False
	
	if (not np.issubdtype(matrix_to_test.dtype, np.floating)):
		return False
	
	if (matrix_to_test.shape != (3, 3)):
		return False
	
	if (not np.allclose(np.dot(matrix_to_test, np.transpose(matrix_to_test)), np.eye(3, 3, dtype=np.float64), rtol=tolerance, atol=tolerance)):
		return False
	
	if (np.abs(np.linalg.det(matrix_to_test) - 1.0) > tolerance):
		return False
	
	return True


def rotation_to_xform_matrix(rotation_matrix):
	assert is_rotation_matrix(rotation_matrix)

	xform_matrix = np.eye(4, 4, dtype=np.float32)
	xform_matrix[0:3, 0:3] = rotation_matrix

	return xform_matrix


def translation_to_xform_matrix(translation_vector):
	assert is_3d_point(translation_vector)

	xform_matrix = np.eye(4, 4, dtype=np.float32)
	xform_matrix[0:3, 3] = translation_vector

	return xform_matrix


def r_and_t_to_xform_matrix(rotation_matrix, translation_vector):
	assert is_rotation_matrix(rotation_matrix)
	assert is_3d_point(translation_vector)

	xform_matrix = np.eye(4, 4, dtype=np.float32)
	xform_matrix[0:3, 0:3] = rotation_matrix
	xform_matrix[0:3, 3] = translation_vector

	return xform_matrix


def vector_to_hat_matrix(vector):
	"""
	If u is a 3D vector, then u-hat is a 3x3 matrix, such that (x-hat) * v is the 
	cross product between u and v.
	"""
	assert isinstance(vector, np.ndarray)
	assert np.issubdtype(vector.dtype, np.floating)
	assert (vector.shape == (3, ))

	return np.array([
		[0.0, -vector[2], vector[1]], 
		[vector[2], 0.0, -vector[0]], 
		[-vector[1], vector[0], 0.0]
	], dtype=vector.dtype)


#############################
# Rotation Matrix Utilities
#############################

def rotation_matrix_to_axis_angle(rotation_matrix, tolerance=1e-4):
	"""
	References
		https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis-angle
	"""

	assert is_rotation_matrix(rotation_matrix)
	tolerance = float(tolerance)
	assert (tolerance > 0.0)

	eigen_value_array, eigen_vector_matrix = np.linalg.eig(rotation_matrix)
	unit_eigen_value_index_array = np.where(np.abs(eigen_value_array - 1.0) < tolerance)[0]
	if (len(unit_eigen_value_index_array) == 0):
		raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
	axis_vector = np.squeeze(np.real(eigen_vector_matrix[:, unit_eigen_value_index_array[0]]))

	cos_angle = (np.trace(rotation_matrix) - 1.0) / 2.0

	# A unit vector of length 3 must have at least 1 component > 0.5
	if (abs(axis_vector[2]) > 0.5):
		sin_angle = (rotation_matrix[1, 0] + (cos_angle - 1.0) * axis_vector[0] * axis_vector[1]) / axis_vector[2]
	elif (abs(axis_vector[1]) > 0.5):
		sin_angle = (rotation_matrix[0, 2] + (cos_angle - 1.0) * axis_vector[0] * axis_vector[2]) / axis_vector[1]
	elif (abs(axis_vector[0]) > 0.5):
		sin_angle = (rotation_matrix[2, 1] + (cos_angle - 1.0) * axis_vector[1] * axis_vector[2]) / axis_vector[0]
	else:
		raise Exception("Bug in rotation_matrix_to_axis_angle: axis_vector must have at least one component > 0.5")

	angle_rad = np.arctan2(sin_angle, cos_angle)
	if (angle_rad < 0.0):
		angle_rad = angle_rad + (2.0 * np.pi)
	if (angle_rad > np.pi):
		angle_rad = (2 * np.pi) - angle_rad
		axis_vector = (-1.0) * axis_vector

	return axis_vector, angle_rad


def axis_angle_to_rotation_matrix(axis_vector, angle_rad):
	"""
	References
		https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
	"""

	assert isinstance(axis_vector, np.ndarray)
	assert np.issubdtype(axis_vector.dtype, np.floating)
	assert (axis_vector.shape == (3, ))

	axis_vector_norm = np.linalg.norm(axis_vector, ord=2, axis=None, keepdims=False)
	assert (axis_vector_norm > 1e-6)
	axis_vector_unit = axis_vector / axis_vector_norm

	axis_cross_product_matrix = vector_to_hat_matrix(axis_vector_unit)

	rotation_matrix = (
		np.eye(3, 3, dtype=np.float64) + (np.sin(angle_rad) * axis_cross_product_matrix) + 
		((1.0 - np.cos(angle_rad)) * np.dot(axis_cross_product_matrix, axis_cross_product_matrix))
	)

	return rotation_matrix


def rotation_matrix_to_viewpoint_angles(camera_to_other_rotation_matrix):
	assert is_rotation_matrix(camera_to_other_rotation_matrix)

	# "Viewpoint" is defined by the direction of camera's (0, 0, -1) vector in the cad reference frame
	camera_frame_point1 = np.array([0.0, 0.0, -1.0], dtype=np.float64)
	cad_frame_point1 = np.reshape(np.dot(camera_to_other_rotation_matrix, np.reshape(camera_frame_point1, (3, 1))), (3, ))

	# Theoretically there should be no need to normalize cad_frame_point1. However, small numerical variations may
	# cause it to have a norm slightly greater than 1, which would cause arcsin and arccos to return nan.
	# Therefore, normalize it to have a norm slightly less than 1 to be safe.
	cad_frame_point1 = cad_frame_point1 / (np.linalg.norm(cad_frame_point1, axis=None) + 1e-8)

	# Calculate the azimuth angle (between 0 and 2*pi)
	azimuth_rad = np.arctan2(cad_frame_point1[1], cad_frame_point1[0])
	if (azimuth_rad < 0.0):
		azimuth_rad = azimuth_rad + (2 * np.pi)

	# Calculate the elevation angle (between -pi/2 and pi/2)
	elevation_rad = np.arcsin(cad_frame_point1[2])

	# Point1 is invariant to axial rotation of the camera. Inspect another point to determine axial rotation.
	camera_frame_point2 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
	cad_frame_point2 = np.reshape(np.dot(camera_to_other_rotation_matrix, np.reshape(camera_frame_point2, (3, 1))), (3, ))
	
	# From azimuth and elevation, directly compute the camera's x and y direction in world frame if axial is 0 
	zero_axial_x_direction = np.array([-np.sin(azimuth_rad), np.cos(azimuth_rad), 0.0], dtype=np.float64)
	zero_axial_y_direction = np.array([np.sin(elevation_rad) * np.cos(azimuth_rad), np.sin(elevation_rad) * np.sin(azimuth_rad), -np.cos(elevation_rad)], dtype=np.float64)

	# Compute the axial rotation angle (between 0 and 2*pi)
	axial_rotation_rad = np.arctan2(np.dot(zero_axial_y_direction, cad_frame_point2), np.dot(zero_axial_x_direction, cad_frame_point2))
	if (axial_rotation_rad < 0.0):
		axial_rotation_rad = axial_rotation_rad + (2 * np.pi)
	
	return (azimuth_rad, elevation_rad, axial_rotation_rad)


def viewpoint_angles_to_rotation_matrix(azimuth_rad, elevation_rad, axial_rotation_rad):
	assert (isinstance(azimuth_rad, float) or isinstance(azimuth_rad, np.floating))
	assert (isinstance(elevation_rad, float) or isinstance(elevation_rad, np.floating))
	assert (isinstance(axial_rotation_rad, float) or isinstance(axial_rotation_rad, np.floating))

	zero_axial_x_direction = np.array([-np.sin(azimuth_rad), np.cos(azimuth_rad), 0.0], dtype=np.float64)
	zero_axial_y_direction = np.array([np.sin(elevation_rad) * np.cos(azimuth_rad), np.sin(elevation_rad) * np.sin(azimuth_rad), -np.cos(elevation_rad)], dtype=np.float64)

	camera_unit_x_mapped_direction = (np.cos(axial_rotation_rad) * zero_axial_x_direction) + (np.sin(axial_rotation_rad) * zero_axial_y_direction)
	camera_unit_y_mapped_direction = (np.cos(axial_rotation_rad) * zero_axial_y_direction) - (np.sin(axial_rotation_rad) * zero_axial_x_direction)
	camera_unit_z_mapped_direction = (-1.0) * np.array([np.cos(elevation_rad) * np.cos(azimuth_rad), np.cos(elevation_rad) * np.sin(azimuth_rad), np.sin(elevation_rad)])

	camera_to_other_rotation_matrix = np.transpose(np.array([camera_unit_x_mapped_direction, camera_unit_y_mapped_direction, camera_unit_z_mapped_direction]))

	return camera_to_other_rotation_matrix


def rotation_matrix_geodesic(rotation_matrix1, rotation_matrix2):
	assert is_rotation_matrix(rotation_matrix1)
	assert is_rotation_matrix(rotation_matrix2)

	relative_rotation_matrix = np.dot(np.transpose(rotation_matrix1), rotation_matrix2)
	(log_relative_rotation_matrix, estimated_error) = scipy.linalg.logm(relative_rotation_matrix, disp=False)
	
	geodesic_distance = np.linalg.norm(log_relative_rotation_matrix, ord="fro", axis=None, keepdims=False) / np.sqrt(2.0)

	return geodesic_distance


def get_point_centering_rotation_matrix(cam1_to_other_rotation_matrix, cam1_point):
	assert is_rotation_matrix(cam1_to_other_rotation_matrix)
	assert is_3d_point(cam1_point)

	tolerance = 1e-4

	cam1_point_norm = np.linalg.norm(cam1_point, ord=2, axis=None, keepdims=False)
	assert (cam1_point_norm > tolerance)
	cam1_point_unit = cam1_point / cam1_point_norm
	
	cam2_point_unit = np.array([0.0, 0.0, 1.0], dtype=np.float64)

	cam1_to_cam2_rotation_axis = np.cross(cam1_point_unit, cam2_point_unit)
	cam1_to_cam2_rotation_axis_norm = np.linalg.norm(cam1_to_cam2_rotation_axis, ord=2, axis=None, keepdims=False)
	if (cam1_to_cam2_rotation_axis_norm < tolerance):
		cam1_to_cam2_rotation_matrix = np.eye(3, 3, dtype=np.float64)
	else:
		relative_rotation_angle_rad = np.arccos(np.dot(cam1_point_unit, cam2_point_unit))
		cam1_to_cam2_rotation_matrix = axis_angle_to_rotation_matrix(cam1_to_cam2_rotation_axis, relative_rotation_angle_rad)
	
	cam2_to_other_rotation_matrix = np.dot(cam1_to_other_rotation_matrix, np.transpose(cam1_to_cam2_rotation_matrix))

	# There is a family of rotations such that cam1_point in cam1 is [0, 0, 1] in cam2.
	# More specifically, the family is (cam2_to_other_rotation_matrix) * (any rotation around z-axis)
	# We arbitrarily pick the one that preserves axial rotation.
	(_, _, old_axial_rotation_rad) = rotation_matrix_to_viewpoint_angles(cam1_to_other_rotation_matrix)
	(_, _, new_axial_rotation_rad) = rotation_matrix_to_viewpoint_angles(cam2_to_other_rotation_matrix)
	axial_rotation_diff_rad = old_axial_rotation_rad - new_axial_rotation_rad
	modifier_rotation_matrix = axis_angle_to_rotation_matrix(np.array([0.0, 0.0, 1.0], dtype=np.float64), axial_rotation_diff_rad)
	
	#modified_cam2_to_other_rotation_matrix = np.dot(cam2_to_other_rotation_matrix, modifier_rotation_matrix)
	modified_cam1_to_cam2_rotation_matrix = np.dot(np.transpose(modifier_rotation_matrix), cam1_to_cam2_rotation_matrix)

	return modified_cam1_to_cam2_rotation_matrix


# ==== Code for visualizing rotation matrix -> viewpoint angle calculation. ====
# import matplotlib.pyplot
# from mpl_toolkits.mplot3d import Axes3D
# figure = matplotlib.pyplot.figure()
# axes = figure.add_subplot(111, projection="3d")
# axes.quiver(
# 	np.cos(elevation_rad) * np.cos(azimuth_rad), np.cos(elevation_rad) * np.sin(azimuth_rad), np.sin(elevation_rad), 
# 	[zero_axial_x_direction[0], zero_axial_y_direction[0]], 
# 	[zero_axial_x_direction[1], zero_axial_y_direction[1]],
# 	[zero_axial_x_direction[2], zero_axial_y_direction[2]],
# 	pivot="tail"
# )
# axes.set_xlim([-2, 2])
# axes.set_xlabel("X")
# axes.set_ylim([-2, 2])
# axes.set_ylabel("Y")
# axes.set_zlim([-2, 2])
# axes.set_zlabel("Z")
# matplotlib.pyplot.show()


#############################
# Quarternion Utilities
#############################

def is_quaternion(q):
	if (not isinstance(q, np.ndarray)):
		return False
	if (not np.issubdtype(q.dtype, np.floating)):
		return False
	if (q.shape != (4, )):
		return False
	return True


def quaternion_norm(q):
	assert is_quaternion(q)
	return np.linalg.norm(q, ord=2, axis=None)


def normalize_quaternion(q):
	assert is_quaternion(q)

	q_norm = quaternion_norm(q)
	assert (q_norm > 1e-6)

	return q / q_norm


def quaternion_product(q1, q2):
	assert is_quaternion(q1)
	assert is_quaternion(q2)
	assert (q1.dtype == q2.dtype)

	q3 = np.copy(q1)
	q3[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
	q3[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
	q3[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
	q3[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
	
	return q3


def quaternion_inverse(q):
	assert is_quaternion(q)

	q_norm = quaternion_norm(q)
	assert (q_norm > 1e-6)
	
	q_inverse = np.copy(q)
	q_inverse[1] = -q[1]
	q_inverse[2] = -q[2]
	q_inverse[3] = -q[3]
	q_inverse = ((q_inverse / q_norm) / q_norm)

	return q_inverse


def apply_quaternion_rotation(rotation_q, point):
	assert is_quaternion(rotation_q)
	assert is_3d_point(point)

	point_q = np.concatenate([np.array([0.0], dtype=point.dtype), point], axis=0)
	rotation_inverse_q = quaternion_inverse(rotation_q)
	point_after_rotation_q = quaternion_product(quaternion_product(rotation_q, point_q), rotation_inverse_q)
	point_after_rotation = point_after_rotation_q[1:]
	
	return point_after_rotation


def quaternion_to_rotation_matrix(q):
	'''
	Calculate rotation matrix corresponding to quaternion.
	Rotation matrix applies to column vectors, and is applied to the left of coordinate vectors. 
	The algorithm here allows quaternions that have not been normalized.

	References
		http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
	'''

	assert is_quaternion(q)
	q_unit = normalize_quaternion(q)
	
	(w, x, y, z) = q_unit

	s = 2.0
	X = x * s
	Y = y * s
	Z = z * s
	wX = w * X; wY = w * Y; wZ = w * Z
	xX = x * X; xY = x * Y; xZ = x * Z
	yY = y * Y; yZ = y * Z; zZ = z * Z

	return np.array([
		[1.0 - (yY + zZ), xY - wZ, xZ + wY], 
		[xY + wZ, 1.0 - (xX + zZ), yZ - wX], 
		[xZ - wY, yZ + wX, 1.0 - (xX + yY)]
	], dtype=np.float64)


def rotation_matrix_to_quaternion(rotation_matrix):
	'''
	Calculate quaternion corresponding to given rotation matrix.
	
	Method claimed to be robust to numerical errors in rotation_matrix.
	Constructs quaternion by calculating maximum eigenvector for matrix K (constructed from input "rotation_matrix"). 
	Although this is not tested, a maximum eigenvalue of 1 corresponds to a valid rotation.
	A quaternion q*-1 corresponds to the same rotation as q; thus the sign of the reconstructed quaternion is arbitrary, 
	and we return quaternions with positive w (q[0]).

	References
		http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
	'''
	
	assert is_rotation_matrix(rotation_matrix)
	
	# Qyx refers to the contribution of the y input vector component to the x output vector component. 
	# Qyx is therefore the same as rotation_matrix[0,1].  The notation is from the Wikipedia article.
	Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = rotation_matrix.flat

	# Fill only lower half of symmetric matrix
	K = np.array([
		[Qxx - Qyy - Qzz, 0,               0,               0              ],
		[Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
		[Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]
	], dtype=np.float64) / 3.0

	# Use Hermitian eigenvectors, values for speed
	eigen_value_array, eigen_vector_matrix = np.linalg.eigh(K, UPLO='L')

	# Select largest eigenvector, reorder to w,x,y,z quaternion
	q = eigen_vector_matrix[[3, 0, 1, 2], np.argmax(eigen_value_array)]

	# Prefer quaternion with positive w, since -q corresponds to same rotation as q
	if q[0] < 0:
		q = (-1.0) * q
	
	return q