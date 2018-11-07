from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
import argparse
import itertools
import numpy as np
import torch.autograd
import torch.nn
import torch.optim.lr_scheduler
import torch.utils.data

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, ".."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import global_variables as global_vars
import util.pytorch
import data.object_class
import data.pytorch_wrapper
import learning.dataset
import learning.network


def construct_argument_parser():
	arg_parser = argparse.ArgumentParser(
		prog="train.py", description="Train the viewpoint network."
	)
	arg_parser.add_argument(
		"--box_perturb", type=float, default=0.15, help="Ratio of perturbation for bounding boxes."
	)
	arg_parser.add_argument(
		"--batch_size", type=int, default=20, help="Number of models in a batch"
	)
	arg_parser.add_argument(
		"--num_workers", type=int, default=4, help="Number of workers for the DataLoader"
	)
	arg_parser.add_argument(
		"--num_epochs", type=int, default=20, help="number of epochs to train"
	)
	arg_parser.add_argument(
		"--learning_rate", type=float, default=0.001, help="learning rate"
	)
	arg_parser.add_argument(
		"--momentum", type=float, default=0.9, help="SGD momentum"
	)
	arg_parser.add_argument(
		"--weight_decay", type=float, default=0.0001, help="rate of regularization loss"
	)
	arg_parser.add_argument(
		"--angle_std_deg", type=float, default=15.0, help="variance of angle in ground truth"
	)
	arg_parser.add_argument(
		"--print_interval", type=int, default=10, help="number of batches between printing messages"
	)
	arg_parser.add_argument(
		"--ckpt_save_dir", type=str, default="log/checkpoint", help="directory to save checkpoint to"
	)
	arg_parser.add_argument(
		"--ckpt_resume_path", type=str, default="", help="path of checkpoint file to load"
	)

	return arg_parser


def main():
	assert torch.cuda.is_available()
	arg_parser = construct_argument_parser()
	arg_namespace = arg_parser.parse_args()

	# Construct dataset and dataloader
	places2_val_dataset = data.pytorch_wrapper.ImageDirDataset(global_vars.PLACES2_VAL_DIR, False)
	occlusion_config = util.image.RandomOcclusionConfig()
	occlusion_config.expansion_ratio = arg_namespace.box_perturb
	rs_augmented_dataset = data.pytorch_wrapper.construct_rs_augmented_dataset(data.object_class.MainClass.CAR, places2_val_dataset, occlusion_config)
	train_dataset = learning.dataset.ViewpointDataset(rs_augmented_dataset, arg_namespace.box_perturb)
	train_dataloader = torch.utils.data.DataLoader(
		train_dataset, batch_size=arg_namespace.batch_size, shuffle=True, num_workers=arg_namespace.num_workers, pin_memory=True
	)

	# Construct network
	viewpoint_network = learning.network.ViewpointNetwork(
		3, global_vars.CNN_VIEWPOINT_INPUT_SIZE, global_vars.CNN_VIEWPOINT_INPUT_SIZE
	)
	viewpoint_network.train()
	viewpoint_network.cuda()

	# Construct Optimizer
	optimizer = torch.optim.SGD(
		viewpoint_network.parameters(), lr=arg_namespace.learning_rate, 
		momentum=arg_namespace.momentum, weight_decay=arg_namespace.weight_decay, 
		nesterov=True
	)

	# Create checkpoint directory
	if os.path.isdir(arg_namespace.ckpt_save_dir):
		user_input = input("\nCheckpoint directory %s already exists! Do you want to delete it? (y/n): " % (arg_namespace.ckpt_save_dir, )).lower()
		if (user_input == "y"):
			shutil.rmtree(arg_namespace.ckpt_save_dir)
		else:
			exit()
	os.makedirs(arg_namespace.ckpt_save_dir)

	# Load saved checkpoint
	saved_epoch_index = -1
	if os.path.isfile(arg_namespace.ckpt_resume_path):
		print("\nLoading checkpoint from %s\n" % (arg_namespace.ckpt_resume_path, ))
		loaded_checkpoint = util.pytorch.TorchCheckpoint()
		loaded_checkpoint.populate_from_dict(torch.load(arg_namespace.ckpt_resume_path))
		assert (loaded_checkpoint.epoch_index is not None)
		assert (loaded_checkpoint.model_state_dict is not None)
		assert (loaded_checkpoint.optimizer_state_dict is not None)
		saved_epoch_index = loaded_checkpoint.epoch_index
		viewpoint_network.load_state_dict(loaded_checkpoint.model_state_dict)
		optimizer.load_state_dict(loaded_checkpoint.optimizer_state_dict)

	learning_rate_scheduler = torch.optim.lr_scheduler.MultiStepLR(
		optimizer, [3, 5, 7, 9, 11, 13], gamma=0.5, last_epoch=saved_epoch_index
	)
	# learning_rate_scheduler = torch.optim.lr_scheduler.MultiStepLR(
	# 	optimizer, [10], gamma=0.1, last_epoch=saved_epoch_index
	# )

	# Main loop
	for epoch_index in range((saved_epoch_index + 1), arg_namespace.num_epochs):
		learning_rate_scheduler.step()

		for (batch_index, batch_data) in enumerate(train_dataloader):
			(batch_input_image_tensor, batch_gt_viewpoint_tensor) = batch_data
			batch_input_image_variable = torch.autograd.Variable(batch_input_image_tensor.cuda(), requires_grad=False)
			batch_gt_viewpoint_variable = torch.autograd.Variable(batch_gt_viewpoint_tensor.cuda(), requires_grad=False)
			batch_gt_azimuth_variable = batch_gt_viewpoint_variable[:, 0]

			# Forward pass
			optimizer.zero_grad()
			batch_azimuth_logit_variable = viewpoint_network(batch_input_image_variable)

			# Compute loss
			azimuth_sample_variable = torch.autograd.Variable(viewpoint_network.azimuth_sample_tensor.cuda(), requires_grad=False)
			batch_azimuth_diff_variable = azimuth_sample_variable.view(1, -1) - torch.unsqueeze(batch_gt_azimuth_variable, 1)
			batch_azimuth_remainder_variable = torch.remainder(batch_azimuth_diff_variable, (2.0 * np.pi))
			batch_azimuth_periodic_dist_variable = torch.min(
				batch_azimuth_remainder_variable, ((2.0 * np.pi) - batch_azimuth_remainder_variable)
			)
			gt_kappa = float(1.0 / np.square(np.deg2rad(arg_namespace.angle_std_deg)))
			batch_gt_azimuth_logit_variable = gt_kappa * torch.cos(batch_azimuth_periodic_dist_variable)
			kl_divergence_variable = util.pytorch.kl_div_discrete_with_logits(
				batch_azimuth_logit_variable, batch_gt_azimuth_logit_variable, 1, False
			)
			loss_variable = torch.mean(kl_divergence_variable)

			# Backward pass
			loss_variable.backward()
			optimizer.step()

			if ((batch_index % arg_namespace.print_interval) == 0):
				print("Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tLR: {:.6f}".format(
					epoch_index, batch_index * arg_namespace.batch_size, len(train_dataset), 
					loss_variable.data[0], learning_rate_scheduler.get_lr()[0]
				))
		
		saved_checkpoint = util.pytorch.TorchCheckpoint()
		saved_checkpoint.epoch_index = epoch_index
		saved_checkpoint.model_state_dict = viewpoint_network.state_dict()
		saved_checkpoint.optimizer_state_dict = optimizer.state_dict()

		saved_checkpoint_path = os.path.join(arg_namespace.ckpt_save_dir, "{}.ckpt".format(epoch_index))
		torch.save(saved_checkpoint.dict_form, saved_checkpoint_path)


if __name__ == "__main__":
	main()