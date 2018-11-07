from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import enum
import warnings
import numpy as np
import xml.etree.ElementTree

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, ".."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import util.transform
import data.object_class


@enum.unique
class TrackletPoseState(enum.IntEnum):
	UNSET = 0
	INTERP = 1
	LABELED = 2


@enum.unique
class TrackletOccState(enum.IntEnum):
	OCCLUSION_UNSET = -1
	VISIBLE = 0
	PARTLY = 1
	FULLY = 2


@enum.unique
class TrackletTruncState(enum.IntEnum):
	TRUNCATION_UNSET = -1
	IN_IMAGE = 0
	TRUNCATED = 1
	OUT_IMAGE = 2
	BEHIND_IMAGE = 99


class TrackletPose(object):
	"""
	Represents the pose of a 3D KITTI tracklet in a particular frame.
	"""

	def __init__(self):
		self._tx = None
		self._ty = None
		self._tz = None
		self._rx = None
		self._ry = None
		self._rz = None
		self._pose_state = None
		self._occlusion_state = None
		self._is_occlusion_keyframe = None
		self._truncation_state = None
	

	def __repr__(self):
		return (
			"TrackletPose(\n"
			"    tx=%.4f, ty=%.4f, tz=%.4f, \n"
			"    rx=%.4f, ry=%.4f, rz=%.4f, \n"
			"    pose_state=%r, \n"
			"    occlusion_state=%r, \n"
			"    is_occlusion_keyframe=%r, \n"
			"    truncation_state=%r\n"
			")"  % (
				self.tx, self.ty, self.tz, self.rx, self.ry, self.rz, self.pose_state, 
				self.occlusion_state, self.is_occlusion_keyframe, self.truncation_state
			)
		)
	

	@property
	def tx(self):
		assert (self._tx is not None)
		return self._tx
	

	@property
	def ty(self):
		assert (self._ty is not None)
		return self._ty
	

	@property
	def tz(self):
		assert (self._tz is not None)
		return self._tz
	

	@property
	def translation_vector(self):
		return np.array([self.tx, self.ty, self.tz], dtype=np.float32)
	

	@property
	def rx(self):
		assert (self._rx is not None)
		return self._rx
	

	@property
	def ry(self):
		assert (self._ry is not None)
		return self._ry
	

	@property
	def rz(self):
		assert (self._rz is not None)
		return self._rz
	

	@property
	def rotation_matrix(self):
		"""
		It is not clear from the KITT documentation what rx, ry, and rz represent.
		Do they represent intrinsic yaw-pitch-roll, or extrinsic yaw-pitch-roll, or some other form of Euler angles?
		I suspect they left this ambiguity because they assume all objects are aligned to gravity.
		Only rz is non-zero in all of their tracklet labelings, so the exact meanings of rx and ry don't matter.
		"""
		# This will not work if rx and ry are nonzero.
		assert (self.rx < 1e-4)
		assert (self.ry < 1e-4)

		# Compute the rotation around z.
		z_rotation_matrix = np.array([
			[np.cos(self.rz), -np.sin(self.rz), 0.0], 
			[np.sin(self.rz), np.cos(self.rz), 0.0], 
			[0.0, 0.0, 1.0]
		], dtype=np.float32)

		return z_rotation_matrix
	

	@property
	def velo_forward_vector(self):
		# Velodyne coordinate system is x forward, y left, z up.
		canonical_forward_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
		return np.matmul(self.rotation_matrix, canonical_forward_vector)
	

	@property
	def pose_state(self):
		assert (self._pose_state is not None)
		return self._pose_state
	

	@property
	def occlusion_state(self):
		assert (self._occlusion_state is not None)
		return self._occlusion_state
	

	@property
	def is_occlusion_keyframe(self):
		assert (self._is_occlusion_keyframe is not None)
		return self._is_occlusion_keyframe
	

	@property
	def truncation_state(self):
		assert (self._truncation_state is not None)
		return self._truncation_state
	

	def load_xml_element(self, tracklet_pose_element):
		assert isinstance(tracklet_pose_element, xml.etree.ElementTree.Element)

		tag_to_child_dict = {
			"tx": None, "ty": None, "tz": None, "rx": None, "ry": None, "rz": None, 
			"state": None, "occlusion": None, "occlusion_kf": None, "truncation": None, 
			"amt_occlusion": None, "amt_occlusion_kf": None, "amt_border_l": None, 
			"amt_border_r": None, "amt_border_kf": None
		}

		for child_element in tracklet_pose_element:
			assert (child_element.tag in tag_to_child_dict)
			assert (tag_to_child_dict[child_element.tag] is None)
			tag_to_child_dict[child_element.tag] = child_element
		
		assert (tag_to_child_dict["tx"] is not None)
		self._tx = float(tag_to_child_dict["tx"].text)

		assert (tag_to_child_dict["ty"] is not None)
		self._ty = float(tag_to_child_dict["ty"].text)

		assert (tag_to_child_dict["tz"] is not None)
		self._tz = float(tag_to_child_dict["tz"].text)

		assert (tag_to_child_dict["rx"] is not None)
		self._rx = float(tag_to_child_dict["rx"].text)

		assert (tag_to_child_dict["ry"] is not None)
		self._ry = float(tag_to_child_dict["ry"].text)

		assert (tag_to_child_dict["rz"] is not None)
		self._rz = float(tag_to_child_dict["rz"].text)

		assert (tag_to_child_dict["state"] is not None)
		self._pose_state = TrackletPoseState(int(tag_to_child_dict["state"].text))

		assert (tag_to_child_dict["occlusion"] is not None)
		self._occlusion_state = TrackletOccState(int(tag_to_child_dict["occlusion"].text))

		assert (tag_to_child_dict["occlusion_kf"] is not None)
		self._is_occlusion_keyframe = bool(int(tag_to_child_dict["occlusion_kf"].text))

		assert (tag_to_child_dict["truncation"] is not None)
		self._truncation_state = TrackletTruncState(int(tag_to_child_dict["truncation"].text))

		# Ignore the Amazon Mechanical Turk annotations for now.
	

	def copy(self):
		copied_tracklet_pose = TrackletPose()

		copied_tracklet_pose._tx = self._tx
		copied_tracklet_pose._ty = self._ty
		copied_tracklet_pose._tz = self._tz
		copied_tracklet_pose._rx = self._rx
		copied_tracklet_pose._ry = self._ry
		copied_tracklet_pose._rz = self._rz
		copied_tracklet_pose._pose_state = self._pose_state
		copied_tracklet_pose._occlusion_state = self._occlusion_state
		copied_tracklet_pose._is_occlusion_keyframe = self._is_occlusion_keyframe
		copied_tracklet_pose._truncation_state = self._truncation_state

		return copied_tracklet_pose


class Tracklet(object):
	"""
	Represents a 3D KITTI tracklet.
	"""

	def __init__(self):
		self._kitti_class_enum = None
		self._height = None
		self._width = None
		self._length = None
		self._first_frame_index = None
		self._pose_list = None
		self._is_finished = None
	

	def __repr__(self):
		tracklet_repr = (
			"Tracklet(\n"
			"    kitti_class_enum=%r, \n"
			"    height=%.4f, width=%.4f, length=%.4f, \n"
			"    first_frame_index=%r, \n"
			"    pose_list=[\n" % (
				self.kitti_class_enum, 
				self.height, self.width, self.length, 
				self.first_frame_index
			)
		)

		for tracklet_pose in self.pose_list:
			tracklet_pose_repr = repr(tracklet_pose)
			# Could use textwrap.indent here if we give up Python2 compatibility (added in Python 3.3).
			indented_tracklet_pose_repr = "".join([
				"        " + line for line in tracklet_pose_repr.splitlines(True)
			])
			tracklet_repr = tracklet_repr + indented_tracklet_pose_repr + ", \n"
		
		tracklet_repr = tracklet_repr + (
			"    ], \n"
			"    is_finished=%r"
			")" % (
				self.is_finished
			)
		)

		return tracklet_repr
	

	@property
	def kitti_class_enum(self):
		assert (self._kitti_class_enum is not None)
		return self._kitti_class_enum
	

	@property
	def kitti_class_index(self):
		return data.object_class.kitti_enum_to_kitti_index(self._kitti_class_enum)
	

	@property
	def height(self):
		assert (self._height is not None)
		return self._height
	

	@property
	def width(self):
		assert (self._width is not None)
		return self._width
	

	@property
	def length(self):
		assert (self._length is not None)
		return self._length
	

	@property
	def canonical_box_center_vector(self):
		return np.array([0.0, 0.0, (self.height / 2.0)], dtype=np.float32)
	

	@property
	def canonical_box_corner_matrix(self):
		"""
		The bounding box corner coordinates that are not rotated yet.
		"""
		return np.array([
			[self.length / 2.0, self.width / 2.0, 0.0], 
			[self.length / 2.0, -self.width / 2.0, 0.0],
			[-self.length / 2.0, -self.width / 2.0, 0.0], 
			[-self.length / 2.0, self.width / 2.0, 0.0], 
			[self.length / 2.0, self.width / 2.0, self.height], 
			[self.length / 2.0, -self.width / 2.0, self.height],
			[-self.length / 2.0, -self.width / 2.0, self.height], 
			[-self.length / 2.0, self.width / 2.0, self.height]
		], dtype=np.float32)
	

	@property
	def first_frame_index(self):
		assert (self._first_frame_index is not None)
		return self._first_frame_index
	

	@property
	def pose_list(self):
		assert (self._pose_list is not None)
		return self._pose_list
	

	@property
	def num_poses(self):
		return len(self.pose_list)
	

	def velo_box_center_vector(self, pose_index):
		tracklet_pose = self.pose_list[pose_index]
		return (
			np.matmul(tracklet_pose.rotation_matrix, self.canonical_box_center_vector) + 
			tracklet_pose.translation_vector
		)
	

	def velo_box_corner_matrix(self, pose_index):
		tracklet_pose = self.pose_list[pose_index]
		return np.transpose(
			np.matmul(tracklet_pose.rotation_matrix, np.transpose(self.canonical_box_corner_matrix)) + 
			np.reshape(tracklet_pose.translation_vector, (3, 1))
		)
	

	@property
	def is_finished(self):
		assert (self._is_finished is not None)
		return self._is_finished
	

	def load_xml_element(self, tracklet_element):
		assert isinstance(tracklet_element, xml.etree.ElementTree.Element)

		tag_to_child_dict = {
			"objectType": None, "h": None, "w": None, "l": None, 
			"first_frame": None, "poses": None, "finished": None
		}

		for child_element in tracklet_element:
			assert (child_element.tag in tag_to_child_dict)
			assert (tag_to_child_dict[child_element.tag] is None)
			tag_to_child_dict[child_element.tag] = child_element
		
		assert (tag_to_child_dict["objectType"] is not None)
		object_type_text = tag_to_child_dict["objectType"].text
		if (object_type_text == "Person (sitting)"):
			object_type_text = "Sitter"
		self._kitti_class_enum = data.object_class.KittiClass[object_type_text.upper()]

		assert (tag_to_child_dict["h"] is not None)
		self._height = float(tag_to_child_dict["h"].text)

		assert (tag_to_child_dict["w"] is not None)
		self._width = float(tag_to_child_dict["w"].text)

		assert (tag_to_child_dict["l"] is not None)
		self._length = float(tag_to_child_dict["l"].text)

		assert (tag_to_child_dict["first_frame"] is not None)
		self._first_frame_index = int(tag_to_child_dict["first_frame"].text)

		# Handle poses
		assert (tag_to_child_dict["poses"] is not None)
		tag_to_poses_child2_dict = {"count": None, "item_version": None, "item": []}
		for child2_element in tag_to_child_dict["poses"]:
			assert (child2_element.tag in tag_to_poses_child2_dict)
			if (child2_element.tag == "item"):
				tag_to_poses_child2_dict[child2_element.tag].append(child2_element)
			else:
				assert (tag_to_poses_child2_dict[child2_element.tag] is None)
				tag_to_poses_child2_dict[child2_element.tag] = child2_element
		
		assert (tag_to_poses_child2_dict["count"] is not None)
		expected_num_poses = int(tag_to_poses_child2_dict["count"].text)

		self._pose_list = []
		for tracklet_pose_element in tag_to_poses_child2_dict["item"]:
			new_tracklet_pose = TrackletPose()
			new_tracklet_pose.load_xml_element(tracklet_pose_element)
			self._pose_list.append(new_tracklet_pose)
		
		# End of handling poses
		self._is_finished = False
		if (tag_to_child_dict["finished"] is not None):
			self._is_finished = bool(int(tag_to_child_dict["finished"].text))
		
		# Validation
		if (not self.is_finished):
			warnings.warn("Tracklet is not finished, annotations may be incomplete")
		
		if (len(self._pose_list) == 0):
			warnings.warn("Tracklet contains zero frames")
		
		if (len(self._pose_list) != expected_num_poses):
			warnings.warn("Tracklet should have %d frames, but only %d frames were found in annotation" % (
				expected_num_poses, len(self._pose_list)
			))
	

	def copy(self):
		copied_tracklet = Tracklet()

		copied_tracklet._kitti_class_enum = self._kitti_class_enum
		copied_tracklet._height = self._height
		copied_tracklet._width = self._width
		copied_tracklet._length = self._length
		copied_tracklet._first_frame_index = self._first_frame_index

		if (self._pose_list is None):
			copied_tracklet._pose_list = None
		else:
			copied_tracklet._pose_list = []
			for pose_index in range(self.num_poses):
				copied_tracklet._pose_list.append(self._pose_list[pose_index].copy())
		
		copied_tracklet._is_finished = self._is_finished

		return copied_tracklet


def load_tracklets_xml_file(tracklets_xml_file_path):
	tracklets_xml_file_path = str(tracklets_xml_file_path)

	root_xml_tree = xml.etree.ElementTree.parse(tracklets_xml_file_path)
	tracklets_xml_element = root_xml_tree.find("tracklets")
	assert (tracklets_xml_element is not None)

	tracklet_list = []
	expected_num_tracklets = None
	for child_element in tracklets_xml_element:
		if (child_element.tag == "count"):
			assert (expected_num_tracklets is None)
			expected_num_tracklets = int(child_element.text)
		
		elif (child_element.tag == "item_version"):
			pass
		
		elif (child_element.tag == "item"):
			new_tracklet = Tracklet()
			new_tracklet.load_xml_element(child_element)
			tracklet_list.append(new_tracklet)
		
		else:
			raise ValueError("Unexpected tag in tracklets xml element: %s!" % (child_element.tag, ))
	
	# Validation
	assert (expected_num_tracklets is not None)

	if (len(tracklet_list) == 0):
		warnings.warn("Tracklet contains zero frames")
	
	if (len(tracklet_list) != expected_num_tracklets):
		warnings.warn("Expected %d tracklets from xml, but only %d  were found" % (
			expected_num_tracklets, len(tracklet_list)
		))
	
	return tracklet_list


class Calibration(object):
	"""
	Represent a complete KITTI calibration. 
	Notation complies with the KITTI devkit.
	"""
	def __init__(self):
		self._cam_cam_s_matrix = None
		self._cam_cam_k_array = None
		self._cam_cam_d_matrix = None
		self._cam_cam_r_array = None
		self._cam_cam_t_matrix = None
		self._cam_cam_s_rect_matrix = None
		self._cam_cam_r_rect_array = None
		self._cam_cam_p_rect_array = None

		self._velo_cam_r_matrix = None
		self._velo_cam_t_vector = None
	

	@property
	def num_cameras(self):
		return 4
	

	@property
	def cam_cam_s_matrix(self):
		assert (self._cam_cam_s_matrix is not None)
		return self._cam_cam_s_matrix
	
	def cam_cam_s_vector(self, camera_index):
		"""
		Unrectified (2, ) image size for camera_index.
		"""
		camera_index = int(camera_index)
		return self.cam_cam_s_matrix[camera_index, :]
	

	@property
	def cam_cam_k_array(self):
		assert (self._cam_cam_k_array is not None)
		return self._cam_cam_k_array
	
	def cam_cam_k_matrix(self, camera_index):
		"""
		Unrectified (3, 3) intrinsic matrix for camera_index.
		"""
		camera_index = int(camera_index)
		return self.cam_cam_k_array[camera_index, :, :]
	

	@property
	def cam_cam_d_matrix(self):
		assert (self._cam_cam_d_matrix is not None)
		return self._cam_cam_d_matrix
	
	def cam_cam_d_vector(self, camera_index):
		"""
		Unrectified (5, ) distortion vector for camera_index.
		"""
		camera_index = int(camera_index)
		return self.cam_cam_d_matrix[camera_index, :]
	

	@property
	def cam_cam_r_array(self):
		assert (self._cam_cam_r_array is not None)
		return self._cam_cam_r_array
	
	def cam_cam_r_matrix(self, camera_index):
		"""
		Unrectified (3, 3) rotation matrix from camera 0 to camera_index.
		"""
		camera_index = int(camera_index)
		return self.cam_cam_r_array[camera_index, :, :]
	

	@property
	def cam_cam_t_matrix(self):
		assert (self._cam_cam_t_matrix is not None)
		return self._cam_cam_t_matrix
	
	def cam_cam_t_vector(self, camera_index):
		"""
		Unrectified (3, ) translation vector from camera 0 to camera_index.
		"""
		camera_index = int(camera_index)
		return self.cam_cam_t_matrix[camera_index, :]
	

	@property
	def cam_cam_s_rect_matrix(self):
		assert (self._cam_cam_s_rect_matrix is not None)
		return self._cam_cam_s_rect_matrix
	
	def cam_cam_s_rect_vector(self, camera_index):
		"""
		Rectified (2, ) image size for camera_index.
		"""
		camera_index = int(camera_index)
		return self.cam_cam_s_rect_matrix[camera_index, :]
	

	@property
	def cam_cam_r_rect_array(self):
		assert (self._cam_cam_r_rect_array is not None)
		return self._cam_cam_r_rect_array
	
	def cam_cam_r_rect_matrix(self, camera_index):
		"""
		(3, 3) rectifying rotation to make image planes co-planar.
		"""
		camera_index = int(camera_index)
		return self.cam_cam_r_rect_array[camera_index, :, :]
	

	@property
	def cam_cam_p_rect_array(self):
		assert (self._cam_cam_p_rect_array is not None)
		return self._cam_cam_p_rect_array
	
	def cam_cam_p_rect_matrix(self, camera_index):
		"""
		(3, 4) matrix that projects rectified camera 0 coordinates to image coordinates for camera_index.
		"""
		camera_index = int(camera_index)
		return self.cam_cam_p_rect_array[camera_index, :, :]
	

	def cam_cam_t_rect_vector(self, camera_index):
		"""
		The p_rect matrix is what projects from camera 0 rectified coordinates to camera XX image plane.
		Therefore, it can be decomposed into camera intrinsics K for camera XX, and translation from camera 0 to camera XX.
		There is no rotation component by definition of camera rectification.
		
		Rectified (3, ) translation vector from camera 0 to camera_index.
		"""
		camera_index = int(camera_index)
		p_rect_matrix = self.cam_cam_p_rect_matrix(camera_index)
		t_rect_vector = p_rect_matrix[0:3, 3] / np.diag(p_rect_matrix[:, 0:3])
		return t_rect_vector
	

	@property
	def cam_cam_k_rect_matrix(self):
		"""
		The p_rect matrix is what projects from camera 0 rectified coordinates to camera XX image plane.
		Therefore, it can be decomposed into camera intrinsics K for camera XX, and translation from camera 0 to camera XX.
		There is no rotation component by definition of camera rectification.
		
		Rectified (3, 3) intrinsic matrix for camera_index.
		"""
		p00_rect_matrix = self.cam_cam_p_rect_matrix(0)
		k_matrix = p00_rect_matrix[:, 0:3]
		return k_matrix
	

	@property
	def velo_cam00_r_matrix(self):
		"""
		Rotation from Velodyne coordinates to unrectified camera 0 coordinates.
		"""
		assert (self._velo_cam_r_matrix is not None)
		return self._velo_cam_r_matrix
	

	@property
	def velo_cam00_t_vector(self):
		"""
		Translation from Velodyne coordinates to unrectified camera 0 coordinates.
		"""
		assert (self._velo_cam_t_vector is not None)
		return self._velo_cam_t_vector
	

	def velo_cam_xform_matrix(self, camera_index):
		"""
		Rigid transformation from Velodyne coordinates to unrectified camera_index coordinates.
		"""
		velo_cam00_xform_matrix = util.transform.r_and_t_to_xform_matrix(self.velo_cam00_r_matrix, self.velo_cam00_t_vector)
		cam00_cam_xform_matrix = util.transform.translation_to_xform_matrix(self.cam_cam_t_rect_vector(camera_index))
		return np.matmul(cam00_cam_xform_matrix, velo_cam00_xform_matrix)
	

	def velo_image_proj_matrix(self, camera_index):
		"""
		Projection matrix from Velodyne coordinates to image coordinates.
		"""
		return np.matmul(
			self.cam_cam_p_rect_matrix(camera_index), 
			np.matmul(
				util.transform.rotation_to_xform_matrix(self.cam_cam_r_rect_matrix(0)), 
				self.velo_cam_xform_matrix(0)
			)
		)
	

	def load_cam_cam_dict(self, calib_dict):
		assert isinstance(calib_dict, dict)

		self._cam_cam_s_matrix = np.zeros((self.num_cameras, 2), dtype=np.float32)
		self._cam_cam_k_array = np.zeros((self.num_cameras, 3, 3), dtype=np.float32)
		self._cam_cam_d_matrix = np.zeros((self.num_cameras, 5), dtype=np.float32)
		self._cam_cam_r_array = np.zeros((self.num_cameras, 3, 3), dtype=np.float32)
		self._cam_cam_t_matrix = np.zeros((self.num_cameras, 3), dtype=np.float32)
		self._cam_cam_s_rect_matrix = np.zeros((self.num_cameras, 2), dtype=np.float32)
		self._cam_cam_r_rect_array = np.zeros((self.num_cameras, 3, 3), dtype=np.float32)
		self._cam_cam_p_rect_array = np.zeros((self.num_cameras, 3, 4), dtype=np.float32)

		for camera_index in range(4):
			self._cam_cam_s_matrix[camera_index, :] = calib_dict["S_%02d" % (camera_index, )]
			self._cam_cam_k_array[camera_index, :, :] = np.reshape(calib_dict["K_%02d" % (camera_index, )], (3, 3))
			self._cam_cam_d_matrix[camera_index, :] = calib_dict["D_%02d" % (camera_index, )]
			self._cam_cam_r_array[camera_index, :, :] = np.reshape(calib_dict["R_%02d" % (camera_index, )], (3, 3))
			self._cam_cam_t_matrix[camera_index, :] = calib_dict["T_%02d" % (camera_index, )]
			self._cam_cam_s_rect_matrix[camera_index, :] = calib_dict["S_rect_%02d" % (camera_index, )]
			self._cam_cam_r_rect_array[camera_index, :, :] = np.reshape(calib_dict["R_rect_%02d" % (camera_index, )], (3, 3))
			self._cam_cam_p_rect_array[camera_index, :, :] = np.reshape(calib_dict["P_rect_%02d" % (camera_index, )], (3, 4))
		
		# Calibrations should not be changed after loading
		self._cam_cam_s_matrix.flags.writeable = False
		self._cam_cam_k_array.flags.writeable = False
		self._cam_cam_d_matrix.flags.writeable = False
		self._cam_cam_r_array.flags.writeable = False
		self._cam_cam_t_matrix.flags.writeable = False
		self._cam_cam_s_rect_matrix.flags.writeable = False
		self._cam_cam_r_rect_array.flags.writeable = False
		self._cam_cam_p_rect_array.flags.writeable = False
	

	def load_velo_cam_dict(self, calib_dict):
		assert isinstance(calib_dict, dict)

		self._velo_cam_r_matrix = np.copy(np.reshape(calib_dict["R"], (3, 3)))
		self._velo_cam_t_vector = np.copy(calib_dict["T"])

		self._velo_cam_r_matrix.flags.writeable = False
		self._velo_cam_t_vector.flags.writeable = False


def read_calib_file(calib_file_path):
	"""
	Read in a calibration file and parse into a dictionary.
	"""
	calib_dict = {}
	with open(calib_file_path, 'r') as calib_file:
		for line in calib_file.readlines():
			line_key, line_value = line.split(':', 1)
			if (line_key == "calib_time"):
				# It's non-float, and we don't care about calibration time anyway.
				pass 
			else:
				line_value_vector = np.array(line_value.split(), dtype=np.float32)
				calib_dict[line_key] = line_value_vector
	
	return calib_dict


# Demo code

# if __name__ == "__main__":
# 	import matplotlib.pyplot
# 	import skimage.io
# 	import global_variables as global_vars
# 	import util.geometry

# 	# Date folder
# 	kitti_date_folder_name = "2011_09_26"
# 	kitti_date_dir = os.path.join(global_vars.KITTI_ROOT_DIR, kitti_date_folder_name)

# 	# Read calibrations
# 	cam_cam_calib_file_path = os.path.join(kitti_date_dir, "calib_cam_to_cam.txt")
# 	cam_cam_calib_dict = read_calib_file(cam_cam_calib_file_path)
# 	velo_cam_calib_file_path = os.path.join(kitti_date_dir, "calib_velo_to_cam.txt")
# 	velo_cam_calib_dict = read_calib_file(velo_cam_calib_file_path)
# 	kitti_calib = Calibration()
# 	kitti_calib.load_cam_cam_dict(cam_cam_calib_dict)
# 	kitti_calib.load_velo_cam_dict(velo_cam_calib_dict)
# 	velo_image02_proj_matrix = kitti_calib.velo_image_proj_matrix(2)

# 	# Sequence folder
# 	kitti_sequence_folder_name = "2011_09_26_drive_0022_sync"
# 	kitti_sequence_dir = os.path.join(kitti_date_dir, kitti_sequence_folder_name)

# 	# Read tracklets
# 	tracklets_xml_file_path = os.path.join(kitti_sequence_dir, "tracklet_labels.xml")
# 	tracklet_list = load_tracklets_xml_file(tracklets_xml_file_path)

# 	# Image folder
# 	image02_dir = os.path.join(kitti_sequence_dir, "image_02")
# 	image02_data_dir = os.path.join(image02_dir, "data")
# 	assert os.path.isdir(image02_data_dir)

# 	# Loop through frames
# 	image02_data_dir_entry_list = os.listdir(image02_data_dir)
# 	for image02_dir_entry in image02_data_dir_entry_list:
# 		(entry_file_name_base, entry_file_extension) = os.path.splitext(image02_dir_entry)
# 		if (entry_file_extension != ".png"):
# 			continue
		
# 		image_path = os.path.join(image02_data_dir, image02_dir_entry)
# 		if (not os.path.isfile(image_path)):
# 			continue
		
# 		# Read image
# 		rgb_image_uybte_array = skimage.io.imread(image_path)
# 		rgb_image_float_array = skimage.img_as_float(rgb_image_uybte_array)

# 		# Visualization Figure
# 		visualization_figure = matplotlib.pyplot.figure()
# 		visualization_axes = visualization_figure.add_subplot(1, 1, 1)
# 		visualization_axes.imshow(rgb_image_float_array)

# 		# Loop through tracklets
# 		frame_index = int(entry_file_name_base)
# 		for tracklet in tracklet_list:
# 			pose_index = frame_index - tracklet.first_frame_index
# 			if ((pose_index < 0) or (pose_index >= tracklet.num_poses)):
# 				continue
			
# 			tracklet_pose = tracklet.pose_list[pose_index]
# 			if ((tracklet_pose.truncation_state is not TrackletTruncState.IN_IMAGE) and (tracklet_pose.truncation_state is not TrackletTruncState.TRUNCATED)):
# 				continue
			
# 			# Project box corners to image and find bounding rectangle
# 			velo_box_corner_matrix = tracklet.velo_box_corner_matrix(pose_index)
# 			velo_box_corner_homo_matrix = np.concatenate([velo_box_corner_matrix, np.ones((8, 1), dtype=np.float32)], axis=1)
# 			image02_box_corner_homo_matrix = np.transpose(np.matmul(velo_image02_proj_matrix, np.transpose(velo_box_corner_homo_matrix)))
# 			image02_box_corner_matrix = image02_box_corner_homo_matrix[:, 0:2] / image02_box_corner_homo_matrix[:, 2, np.newaxis]
# 			bounding_rectangle = util.geometry.points_to_bounding_rectangle(
# 				image02_box_corner_matrix[:, 0], image02_box_corner_matrix[:, 1]
# 			)

# 			# 2D bounding box
# 			if (tracklet_pose.occlusion_state is TrackletOccState.VISIBLE):
# 				edgecolor="g"
# 			elif (tracklet_pose.occlusion_state is TrackletOccState.PARTLY):
# 				edgecolor="y"
# 			else:
# 				edgecolor="r"
# 			bounding_rectangle_patch = matplotlib.patches.Rectangle(
# 				(bounding_rectangle.min_x, bounding_rectangle.min_y), bounding_rectangle.size_x, bounding_rectangle.size_y, 
# 				linewidth=1, edgecolor=edgecolor, fill=False
# 			)
# 			visualization_axes.add_patch(bounding_rectangle_patch)

# 		matplotlib.pyplot.show()