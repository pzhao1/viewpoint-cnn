from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import torch.autograd
import matplotlib.pyplot
import skimage.io

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, ".."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import global_variables as global_vars
import util.geometry
import util.image
import util.pytorch
import util.statistics
import data.object_class
import data.kitti
import learning.network


def construct_argument_parser():
	arg_parser = argparse.ArgumentParser(
		prog="eval_kiti.py", description="Train the visibility network."
	)
	arg_parser.add_argument(
		"checkpoint_path", help="The path of the checkpoint file to load."
	)
	arg_parser.add_argument(
		"--angle_std_deg", type=float, default=15.0, help="variance of angle in ground truth"
	)

	return arg_parser


def main():
	assert torch.cuda.is_available()
	arg_parser = construct_argument_parser()
	arg_namespace = arg_parser.parse_args()

	# Construct network
	viewpoint_network = learning.network.ViewpointNetwork(
		3, global_vars.CNN_VIEWPOINT_INPUT_SIZE, global_vars.CNN_VIEWPOINT_INPUT_SIZE
	)
	azimuth_sample_tensor = viewpoint_network.azimuth_sample_tensor
	azimuth_sample_vector = azimuth_sample_tensor.numpy()
	viewpoint_network.eval()
	viewpoint_network.cuda()

	# Load checkpoint
	print("\nLoading checkpoint from %s\n" % (arg_namespace.checkpoint_path, ))
	loaded_checkpoint = util.pytorch.TorchCheckpoint()
	loaded_checkpoint.populate_from_dict(torch.load(arg_namespace.checkpoint_path))
	assert (loaded_checkpoint.model_state_dict is not None)
	viewpoint_network.load_state_dict(loaded_checkpoint.model_state_dict)

	# Date folder
	kitti_date_folder_name = "2011_09_26"
	kitti_date_dir = os.path.join(global_vars.KITTI_ROOT_DIR, kitti_date_folder_name)

	# Read calibrations
	cam_cam_calib_file_path = os.path.join(kitti_date_dir, "calib_cam_to_cam.txt")
	cam_cam_calib_dict = data.kitti.read_calib_file(cam_cam_calib_file_path)
	velo_cam_calib_file_path = os.path.join(kitti_date_dir, "calib_velo_to_cam.txt")
	velo_cam_calib_dict = data.kitti.read_calib_file(velo_cam_calib_file_path)
	kitti_calib = data.kitti.Calibration()
	kitti_calib.load_cam_cam_dict(cam_cam_calib_dict)
	kitti_calib.load_velo_cam_dict(velo_cam_calib_dict)
	velo_cam02_xform_matrix = kitti_calib.velo_cam_xform_matrix(2)
	velo_image02_proj_matrix = kitti_calib.velo_image_proj_matrix(2)

	# Sequence folder
	kitti_sequence_folder_name = "2011_09_26_drive_0035_sync"
	kitti_sequence_dir = os.path.join(kitti_date_dir, kitti_sequence_folder_name)
	tracklets_xml_file_path = os.path.join(kitti_sequence_dir, "tracklet_labels.xml")
	tracklet_list = data.kitti.load_tracklets_xml_file(tracklets_xml_file_path)
	image02_dir = os.path.join(kitti_sequence_dir, "image_02")
	image02_data_dir = os.path.join(image02_dir, "data")
	assert os.path.isdir(image02_data_dir)

	# Construct accumulators
	num_total_modes = 4
	error_vector_list = []
	is_visibile_list = []
	is_lightly_occluded_list = []
	is_largely_occluded_list = []

	# Loop through frames
	image02_data_dir_entry_list = sorted(os.listdir(image02_data_dir))
	for image02_dir_entry in image02_data_dir_entry_list:
		(entry_file_name_base, entry_file_extension) = os.path.splitext(image02_dir_entry)
		if (entry_file_extension != ".png"):
			continue
		
		image_path = os.path.join(image02_data_dir, image02_dir_entry)
		if (not os.path.isfile(image_path)):
			continue
		
		# Read image
		kitti_rgb_image_uybte_array = skimage.io.imread(image_path)
		kitti_rgb_image_float_array = skimage.img_as_float(kitti_rgb_image_uybte_array)

		# Loop through tracklets
		frame_index = int(entry_file_name_base)
		for tracklet in tracklet_list:
			kitti_class_index = tracklet.kitti_class_index
			if (not data.object_class.is_kitti_index_handled(kitti_class_index)):
				continue
			
			class_index = data.object_class.kitti_index_to_main_index(kitti_class_index)
			if (class_index != data.object_class.MainClass.CAR):
				continue

			pose_index = frame_index - tracklet.first_frame_index
			if ((pose_index < 0) or (pose_index >= tracklet.num_poses)):
				continue
			
			tracklet_pose = tracklet.pose_list[pose_index]
			if (tracklet_pose.truncation_state is not data.kitti.TrackletTruncState.IN_IMAGE):
				continue
			
			# Project forward direction to camera coordinate
			velo_forward_vector = tracklet_pose.velo_forward_vector
			velo_forward_homo_vector = np.concatenate([velo_forward_vector, np.array([0.0], dtype=np.float32)], axis=0)
			cam02_forward_homo_vector = np.matmul(velo_cam02_xform_matrix, velo_forward_homo_vector)
			cam02_forward_vector = cam02_forward_homo_vector[0:3]

			# Compute object box center in camera coordinate
			velo_box_center_vector = tracklet.velo_box_center_vector(pose_index)
			velo_box_center_homo_vector = np.concatenate([velo_box_center_vector, np.array([1.0], dtype=np.float32)], axis=0)
			cam02_box_center_homo_vector = np.matmul(velo_cam02_xform_matrix, velo_box_center_homo_vector)
			cam02_box_center_vector = cam02_box_center_homo_vector[0:3] / cam02_box_center_homo_vector[3]

			# Angle from positive z direction to object forward direction
			forward_angle = np.arctan2(cam02_forward_vector[0], cam02_forward_vector[2])
			# Angle from positive z direction to object box center direction
			box_center_angle = np.arctan2(cam02_box_center_vector[0], cam02_box_center_vector[2])
			# Ground truth azimuth angle
			gt_azimuth = np.mod((forward_angle - (np.pi / 2.0) - box_center_angle), 2.0 * np.pi)

			# Project box corners to image and find bounding rectangle
			velo_box_corner_matrix = tracklet.velo_box_corner_matrix(pose_index)
			velo_box_corner_homo_matrix = np.concatenate([velo_box_corner_matrix, np.ones((8, 1), dtype=np.float32)], axis=1)
			image02_box_corner_homo_matrix = np.transpose(np.matmul(velo_image02_proj_matrix, np.transpose(velo_box_corner_homo_matrix)))
			image02_box_corner_matrix = image02_box_corner_homo_matrix[:, 0:2] / image02_box_corner_homo_matrix[:, 2, np.newaxis]
			crop_rectangle = util.geometry.points_to_bounding_rectangle(
				image02_box_corner_matrix[:, 0], image02_box_corner_matrix[:, 1]
			)
			image_rectangle = util.geometry.Rectangle(0.0, 0.0, (kitti_rgb_image_float_array.shape[1] - 1.0), (kitti_rgb_image_float_array.shape[0] - 1.0))
			if (not image_rectangle.contains_rectangle(crop_rectangle)):
				continue

			# Construct input to network
			crop_image_index_object = util.image.rectangle_to_image_index(crop_rectangle)
			cropped_rgb_image_float_array = kitti_rgb_image_float_array[crop_image_index_object]
			input_rgb_image_float_array = skimage.transform.resize(
				cropped_rgb_image_float_array, (global_vars.CNN_VIEWPOINT_INPUT_SIZE, global_vars.CNN_VIEWPOINT_INPUT_SIZE), 
				order=1, mode="constant", preserve_range=False, anti_aliasing=True
			)

			# Network forward pass
			batch_input_array = input_rgb_image_float_array[np.newaxis, ...]
			batch_input_tensor = torch.from_numpy(np.transpose(batch_input_array, (0, 3, 1, 2))).float()
			batch_input_variable = torch.autograd.Variable(batch_input_tensor.cuda(), requires_grad=False)
			batch_azimuth_logit_variable = viewpoint_network(batch_input_variable)
			batch_azimuth_logit_matrix = batch_azimuth_logit_variable.data.cpu().numpy()
			batch_azimuth_prob_matrix = util.statistics.softmax(batch_azimuth_logit_matrix, 1)
			azimuth_prob_vector = batch_azimuth_prob_matrix[0, :]

			# Inferecnce: use EM to estimate a von-mises mixture distribution
			train_kappa_value = 1.0 / np.square(np.deg2rad(arg_namespace.angle_std_deg))
			(em_azimuth_mix_vector, em_azimuth_mu_vector, em_azimuth_kappa_vector) = util.statistics.von_mises_mixture_em(
				azimuth_prob_vector, num_total_modes, 0.1, (train_kappa_value * 0.25, train_kappa_value * 4.0), 360, 200
			)
			mix_argsort_array = np.flip(np.argsort(em_azimuth_mix_vector, axis=None, kind="quicksort"), axis=0)
			sorted_azimuth_mu_vector = em_azimuth_mu_vector[mix_argsort_array]

			# Compute difference between inference and GT
			azimuth_mod_vector = np.mod((sorted_azimuth_mu_vector - gt_azimuth), (2.0 * np.pi))
			error_vector = np.minimum(azimuth_mod_vector, ((2.0 * np.pi) - azimuth_mod_vector))
			error_vector_list.append(error_vector)
			is_visibile_list.append(tracklet_pose.occlusion_state is data.kitti.TrackletOccState.VISIBLE)
			is_lightly_occluded_list.append(tracklet_pose.occlusion_state is data.kitti.TrackletOccState.PARTLY)
			is_largely_occluded_list.append(tracklet_pose.occlusion_state is data.kitti.TrackletOccState.FULLY)

			# Visualization
			# visualization_figure = matplotlib.pyplot.figure()
			# image_axes = visualization_figure.add_subplot(2, 1, 1)
			# image_axes.imshow(kitti_rgb_image_float_array)
			# bounding_rectangle_patch = matplotlib.patches.Rectangle(
			# 	(crop_rectangle.min_x, crop_rectangle.min_y), crop_rectangle.size_x, crop_rectangle.size_y, 
			# 	linewidth=1.5, edgecolor="g", fill=False
			# )
			# image_axes.add_patch(bounding_rectangle_patch)
			# image_axes.axis("off")

			# prob_axes = visualization_figure.add_subplot(2, 1, 2)
			# gt_azimuth_deg = np.rad2deg(gt_azimuth)
			# max_azimuth_prob = np.amax(azimuth_prob_vector, axis=None)
			# prob_axes.plot([gt_azimuth_deg, gt_azimuth_deg], [0.0, max_azimuth_prob], "g-", linewidth=3)
			# prob_axes.plot(np.rad2deg(azimuth_sample_vector), azimuth_prob_vector, "b.")
			# for component_index in range(len(em_azimuth_mix_vector)):
			# 	component_mix = em_azimuth_mix_vector[component_index]
			# 	component_mu_deg = np.rad2deg(em_azimuth_mu_vector[component_index])
			# 	prob_axes.plot([component_mu_deg, component_mu_deg], [0.0, component_mix * max_azimuth_prob], 'r--', linewidth=1.5)
		
			# prob_axes.set_xlabel("Azimuth (deg)")
			# prob_axes.set_ylabel("Probability")
			# prob_axes.legend(["True Azimuth", "Output Prob", "Estimated Modes"])

			# matplotlib.pyplot.tight_layout()
			# matplotlib.pyplot.show()
		
		print("Finished frame %s" % (entry_file_name_base, ))
	
	num_eval_examples = len(error_vector_list)
	error_matrix = np.array(error_vector_list, dtype=np.float32)
	is_visible_mask = np.array(is_visibile_list, dtype=np.bool_)
	is_lightly_occluded_mask = np.array(is_lightly_occluded_list, dtype=np.bool_)
	is_largely_occluded_mask = np.array(is_largely_occluded_list, dtype=np.bool_)

	is_error_less15_mask = (error_matrix < (np.pi / 12))
	is_error_less30_mask = (error_matrix < (np.pi / 6))
	for num_counted_modes in range(1, (num_total_modes + 1)):
		positive15_mask = np.any(is_error_less15_mask[:, 0:num_counted_modes], axis=1, keepdims=False)
		positive30_mask = np.any(is_error_less30_mask[:, 0:num_counted_modes], axis=1, keepdims=False)

		positive15_ratio = np.sum(positive15_mask, axis=None, dtype=np.float32) / float(num_eval_examples)
		positive30_ratio = np.sum(positive30_mask, axis=None, dtype=np.float32) / float(num_eval_examples)

		visible_positive15_ratio = (
			np.sum(np.logical_and(positive15_mask, is_visible_mask), axis=None, dtype=np.float32) / 
			np.sum(is_visible_mask, axis=None, dtype=np.float32)
		)
		visible_positive30_ratio = (
			np.sum(np.logical_and(positive30_mask, is_visible_mask), axis=None, dtype=np.float32) / 
			np.sum(is_visible_mask, axis=None, dtype=np.float32)
		)

		lightly_occluded_positive15_ratio = (
			np.sum(np.logical_and(positive15_mask, is_lightly_occluded_mask), axis=None, dtype=np.float32) / 
			np.sum(is_lightly_occluded_mask, axis=None, dtype=np.float32)
		)
		lightly_occluded_positive30_ratio = (
			np.sum(np.logical_and(positive30_mask, is_lightly_occluded_mask), axis=None, dtype=np.float32) / 
			np.sum(is_lightly_occluded_mask, axis=None, dtype=np.float32)
		)

		largely_occluded_positive15_ratio = (
			np.sum(np.logical_and(positive15_mask, is_largely_occluded_mask), axis=None, dtype=np.float32) / 
			np.sum(is_largely_occluded_mask, axis=None, dtype=np.float32)
		)
		largely_occluded_positive30_ratio = (
			np.sum(np.logical_and(positive30_mask, is_largely_occluded_mask), axis=None, dtype=np.float32) / 
			np.sum(is_largely_occluded_mask, axis=None, dtype=np.float32)
		)

		print("Number of modes: %d" % (num_counted_modes, ))
		print("\tOverall: positive15 %0.4f, positive30 %0.4f" % (positive15_ratio, positive30_ratio))
		print("\tVisible: positive15 %0.4f, positive30 %0.4f" % (visible_positive15_ratio, visible_positive30_ratio))
		print("\tLightly Occluded: positive15 %0.4f, positive30 %0.4f" % (lightly_occluded_positive15_ratio, lightly_occluded_positive30_ratio))
		print("\tLargely Occluded: positive15 %0.4f, positive30 %0.4f" % (largely_occluded_positive15_ratio, largely_occluded_positive30_ratio))


if __name__ == "__main__":
	main()