from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import torch.autograd
import torch.nn

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, ".."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import util.pytorch


class ViewpointNetwork(torch.nn.Module):
	"""
	The network that predicts viewpoint from a given image and class.
	Uses the Resnet18 architecture as backbone.

	References:
		https://arxiv.org/abs/1505.05641
		https://arxiv.org/abs/1512.03385
	"""

	def __init__(self, num_input_channels, input_width, input_height):
		# Check arguments
		num_input_channels = int(num_input_channels)
		assert (num_input_channels > 0)
		self._num_input_channels = num_input_channels

		input_width = int(input_width)
		assert (input_width > 0)
		self._input_width = input_width

		input_height = int(input_height)
		assert (input_height > 0)
		self._input_height = input_height

		# Call superclass constructor
		super(ViewpointNetwork, self).__init__()

		# Unit 1 (initial convoluition and pooling)
		self.conv1 = torch.nn.Conv2d(num_input_channels, 64, 7, stride=2, padding=3, bias=False)
		self.bn1 = torch.nn.BatchNorm2d(self.conv1.out_channels)
		self.relu1 = torch.nn.ReLU(inplace=False)
		self.pool1 = torch.nn.MaxPool2d(3, stride=2, padding=1)

		# Unit 2 (conv2_x in Table 1 of reference)
		self.unit2 = util.pytorch.ResnetUnit(2, self.conv1.out_channels, 1, 64)

		# Unit 3 (conv3_x in Table 1 of reference)
		self.unit3 = util.pytorch.ResnetUnit(2, self.unit2.num_output_channels, 2, 128)

		# Unit 4 (conv4_x in Table 1 of reference)
		self.unit4 = util.pytorch.ResnetUnit(2, self.unit3.num_output_channels, 2, 256)

		# Unit 5 (conv5_x in Table 1 of reference)
		self.unit5 = util.pytorch.ResnetUnit(2, self.unit4.num_output_channels, 2, 512)

		# The size of unit5 output is computed by feeding a dummy tensor through the conv layers.
		# Could also use the formula in http://pytorch.org/docs/master/nn.html#conv2d, 
		# but it is more complicated and doesn't provide much speed improvement.
		dummy_input_variable = torch.autograd.Variable(torch.FloatTensor(1, num_input_channels, input_height, input_width), requires_grad=False)
		dummy_unit5_output_variable = self.compute_unit5_output(dummy_input_variable)
		(_, _, self._unit5_output_height, self._unit5_output_width) = dummy_unit5_output_variable.size()

		# FC 6
		self.fc6 = torch.nn.Linear(self.num_fc6_input_features, 1024, bias=False)
		self.bn6 = torch.nn.BatchNorm1d(self.fc6.out_features)
		self.relu6 = torch.nn.ReLU(inplace=False)

		# FC 7
		self.fc7 = torch.nn.Linear(self.fc6.out_features, self.azimuth_sample_tensor.numel(), bias=True)
	

	@property
	def azimuth_sample_tensor(self):
		return torch.linspace(0.0, np.deg2rad(359.0), steps=360)
	

	@property
	def num_input_channels(self):
		return self._num_input_channels
	

	@property
	def input_width(self):
		return self._input_width
	

	@property
	def input_height(self):
		return self._input_height
	

	def compute_unit5_output(self, input_variable):
		assert isinstance(input_variable, torch.autograd.Variable)
		assert (input_variable.dim() == 4)
		(batch_size, num_input_channels, input_height, input_width) = input_variable.size()
		assert (num_input_channels == self.num_input_channels)
		assert (input_width == self.input_width)
		assert (input_height == self.input_height)

		# Standardize input image
		# input_mean_variable = torch.mean(torch.mean(input_variable, 3, keepdim=True), 2, keepdim=True)
		# input_stddev_variable = torch.std(input_variable.view(batch_size, num_input_channels, -1, 1), 2, keepdim=True)
		# min_stddev = float(1.0 / np.sqrt(input_height * input_width))
		# adjusted_stddev_variable = torch.clamp(input_stddev_variable, min=min_stddev)
		# standardized_input_variable = (input_variable - input_mean_variable) / adjusted_stddev_variable

		unit1_output_variable = self.pool1(self.relu1(self.bn1(self.conv1(input_variable))))
		unit2_output_variable = self.unit2(unit1_output_variable)
		unit3_output_variable = self.unit3(unit2_output_variable)
		unit4_output_variable = self.unit4(unit3_output_variable)
		unit5_output_variable = self.unit5(unit4_output_variable)

		return unit5_output_variable
	

	@property
	def num_fc6_input_features(self):
		return (self.unit5.num_output_channels * self._unit5_output_height * self._unit5_output_width)
	

	def forward(self, input_variable):
		uni5_output_variable = self.compute_unit5_output(input_variable)
		(batch_size, num_input_channels, input_height, input_width) = input_variable.size()

		unit6_input_variable = uni5_output_variable.view(batch_size, self.num_fc6_input_features)
		unit6_output_variable = self.relu6(self.bn6(self.fc6(unit6_input_variable)))
		unit7_output_variable = self.fc7(unit6_output_variable)

		return unit7_output_variable