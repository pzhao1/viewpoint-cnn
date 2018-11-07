from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import torch.autograd
import matplotlib.pyplot

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
import data.pytorch_wrapper
import learning.dataset
import learning.network


def construct_argument_parser():
	arg_parser = argparse.ArgumentParser(
		prog="demo_synthetic.py", description="Demo the viewpoint network on synthetic data."
	)
	arg_parser.add_argument(
		"checkpoint_path", help="The path of the checkpoint file to load."
	)
	arg_parser.add_argument(
		"--box_perturb", type=float, default=0.15, help="Ratio of perturbation for bounding boxes."
	)
	arg_parser.add_argument(
		"--angle_std_deg", type=float, default=15.0, help="variance of angle in ground truth"
	)

	return arg_parser


def main():
	assert torch.cuda.is_available()
	arg_parser = construct_argument_parser()
	arg_namespace = arg_parser.parse_args()

	# Construct dataset
	class_enum = data.object_class.MainClass.CAR
	places2_val_dataset = data.pytorch_wrapper.ImageDirDataset(global_vars.PLACES2_VAL_DIR, False)
	occlusion_config = util.image.RandomOcclusionConfig()
	occlusion_config.expansion_ratio = arg_namespace.box_perturb
	rs_augmented_dataset = data.pytorch_wrapper.construct_rs_augmented_dataset(
		class_enum, places2_val_dataset, occlusion_config
	)
	viewpoint_dataset = learning.dataset.ViewpointDataset(rs_augmented_dataset, arg_namespace.box_perturb)

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

	# Main loop
	for dummy_index in range(100):
		item_index = np.random.randint(0, len(viewpoint_dataset), size=None)

		# Read data
		(input_image_tensor, gt_viewpoint_tensor, extra_item_tuple) = viewpoint_dataset.get_item_with_extra(item_index)
		(augmented_datum, crop_rectangle) = extra_item_tuple
		full_rectangle = util.geometry.nonzero_bounding_rectangle(augmented_datum.occlusion_result.full_mask)

		# Pass through network
		batch_input_image_tensor = torch.unsqueeze(input_image_tensor, 0)
		batch_input_image_variable = torch.autograd.Variable(batch_input_image_tensor.cuda(), requires_grad=False)
		batch_azimuth_logit_variable = viewpoint_network(batch_input_image_variable)
		assert (batch_azimuth_logit_variable.size() == (1, azimuth_sample_tensor.numel()))
		azimuth_logit_variable = torch.squeeze(batch_azimuth_logit_variable)
		azimuth_logit_vector = azimuth_logit_variable.data.cpu().numpy()
		azimuth_prob_vector = util.statistics.softmax(azimuth_logit_vector, 0)

		# Inferecnce: use EM to estimate a von-mises mixture distribution
		train_kappa_value = 1.0 / np.square(np.deg2rad(arg_namespace.angle_std_deg))
		(em_azimuth_mix_vector, em_azimuth_mu_vector, em_azimuth_kappa_vector) = util.statistics.von_mises_mixture_em(
			azimuth_prob_vector, 4, 0.1, (train_kappa_value * 0.25, train_kappa_value * 4.0), 360, 200
		)

		# Visualizations
		visualization_figure = matplotlib.pyplot.figure()

		image_axes = visualization_figure.add_subplot(2, 1, 1)
		image_axes.imshow(augmented_datum.augmented_rgb_image_float_array)
		full_rectangle_patch = matplotlib.patches.Rectangle(
			(full_rectangle.min_x, full_rectangle.min_y), full_rectangle.size_x, full_rectangle.size_y, linewidth=2, edgecolor="g", fill=False
		)
		image_axes.add_patch(full_rectangle_patch)
		crop_rectangle_patch = matplotlib.patches.Rectangle(
			(crop_rectangle.min_x, crop_rectangle.min_y), crop_rectangle.size_x, crop_rectangle.size_y, linewidth=1, edgecolor="r", fill=False
		)
		image_axes.add_patch(crop_rectangle_patch)

		prob_axes = visualization_figure.add_subplot(2, 1, 2)
		gt_azimuth_deg = np.rad2deg(gt_viewpoint_tensor[0])
		max_azimuth_prob = np.amax(azimuth_prob_vector, axis=None)
		prob_axes.plot(np.rad2deg(azimuth_sample_vector), azimuth_prob_vector, "b.", label="Prob")
		prob_axes.plot([gt_azimuth_deg, gt_azimuth_deg], [0.0, max_azimuth_prob], "k-", label="GT", linewidth=2)
		for component_index in range(len(em_azimuth_mix_vector)):
			component_mix = em_azimuth_mix_vector[component_index]
			component_mu_deg = np.rad2deg(em_azimuth_mu_vector[component_index])
			prob_axes.plot([component_mu_deg, component_mu_deg], [0.0, component_mix * max_azimuth_prob], 'r-', label="EM", linewidth=1)

		matplotlib.pyplot.show()


if __name__ == "__main__":
	main()
