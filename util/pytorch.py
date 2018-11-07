from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum
from collections import OrderedDict
import torch.nn


def stable_log_sigmoid(input_tensor):
	"""
	sigmoid(x) = exp(x) / (1 + exp(x)) = exp(x) / [exp(max(x, 0)) * (exp(-max(x, 0)) + exp(x - max(x, 0)))]
	log_sigmoid(x) = x - max(x, 0) - log(exp(-max(x, 0)) + exp(x - max(x, 0)))
	"""
	max0_tensor = torch.max(input_tensor, 0.0 * input_tensor)
	log_denominator_tensor = max0_tensor + torch.log(
		torch.exp(-max0_tensor) + torch.exp(input_tensor - max0_tensor)
	)
	log_sigmoid_tensor = input_tensor - log_denominator_tensor

	return log_sigmoid_tensor


def stable_log_sum_exp(input_tensor, dim, keepdim):
	keepdim = bool(keepdim)

	(max_input_tensor, _) = torch.max(input_tensor, dim, keepdim=True)
	log_sum_exp_tensor = max_input_tensor + torch.log(
		torch.sum(torch.exp(input_tensor - max_input_tensor), dim, keepdim=True)
	)

	if keepdim:
		return log_sum_exp_tensor
	else:
		return torch.squeeze(log_sum_exp_tensor, dim=dim)


def stable_log_softmax(input_tensor, dim):
	log_sum_exp_tensor = stable_log_sum_exp(input_tensor, dim, True)
	log_softmax_tensor = input_tensor - log_sum_exp_tensor

	return log_softmax_tensor


def kl_div_discrete_with_logits(input_logit_tensor, target_logit_tensor, dim, keepdim):
	input_log_softmax_tensor = stable_log_softmax(input_logit_tensor, dim)
	target_log_softmax_tensor = stable_log_softmax(target_logit_tensor, dim)
	target_softmax_tensor = torch.exp(target_log_softmax_tensor)

	kl_divergence_tensor = torch.sum(
		(target_softmax_tensor * (target_log_softmax_tensor - input_log_softmax_tensor)), dim, keepdim
	)
	return kl_divergence_tensor


def kl_div_gaussian_with_log_std(mean0_tensor, log_std0_tensor, mean1_tensor, log_std1_tensor):
	"""
	Compute KL( N(mean0, std0) || N(mean1, std1) ), using log standard deviation as inputs.

	References:
		https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
	"""

	std0_tensor = torch.exp(log_std0_tensor)
	std0_squared_tensor = torch.pow(std0_tensor, 2.0)
	std1_tensor = torch.exp(log_std1_tensor)
	std1_squared_tensor = torch.pow(std1_tensor, 2.0)

	kl_div_tensor = 0.5 * (
		(std0_squared_tensor / std1_squared_tensor) + 
		(torch.pow((mean1_tensor - mean0_tensor), 2.0) / std1_squared_tensor) - 1.0 + 
		(2.0 * (log_std1_tensor - log_std0_tensor))
	)

	return kl_div_tensor


def reverse_pixel_shuffle(input_tensor, downsample_rate):
	assert (input_tensor.dim() == 4)
	(batch_size, num_input_channels, input_size_y, input_size_x) = input_tensor.size()

	downsample_rate = int(downsample_rate)
	assert (downsample_rate >= 1)
	assert ((input_size_x % downsample_rate) == 0)
	assert ((input_size_y % downsample_rate) == 0)

	stack_tensor_list = []
	for y_start_index in range(0, downsample_rate):
		for x_start_index in range(0, downsample_rate):
			stack_tensor_list.append(input_tensor[:, :, y_start_index::downsample_rate, x_start_index::downsample_rate])
	
	output_tensor = torch.cat(stack_tensor_list, dim=1)
	return output_tensor


class TorchCheckpointKey(Enum):
	EPOCH = "epoch_index"
	MODEL = "model_state_dict"
	OPTIMIZER = "optimizer_state_dict"
	CUSTOM = "custom_dict"


class TorchCheckpoint(object):

	def __init__(self):
		self.epoch_index = None
		self.model_state_dict = None
		self.optimizer_state_dict = None
		self.custom_dict = None
	

	@property
	def epoch_index(self):
		return self._epoch_index
	
	@epoch_index.setter
	def epoch_index(self, value):
		if value is None:
			self._epoch_index = None
		else:
			value = int(value)
			assert (value >= 0)
			self._epoch_index = value
	

	@property
	def model_state_dict(self):
		return self._model_state_dict
	
	@model_state_dict.setter
	def model_state_dict(self, value):
		if value is None:
			self._model_state_dict = None
		else:
			assert isinstance(value, dict)
			self._model_state_dict = value
	

	@property
	def optimizer_state_dict(self):
		return self._optimizer_state_dict
	
	@optimizer_state_dict.setter
	def optimizer_state_dict(self, value):
		if value is None:
			self._optimizer_state_dict = None
		else:
			assert isinstance(value, dict)
			self._optimizer_state_dict = value
	

	@property
	def custom_dict(self):
		return self._custom_dict
	
	@custom_dict.setter
	def custom_dict(self, value):
		if value is None:
			self._custom_dict = None
		else:
			assert isinstance(value, dict)
			self._custom_dict = value
	

	@property
	def dict_form(self):
		return {
			TorchCheckpointKey.EPOCH.value: self.epoch_index, 
			TorchCheckpointKey.MODEL.value: self.model_state_dict, 
			TorchCheckpointKey.OPTIMIZER.value: self.optimizer_state_dict, 
			TorchCheckpointKey.CUSTOM.value: self.custom_dict
		}
	

	def populate_from_dict(self, checkpoint_dict_form):
		assert isinstance(checkpoint_dict_form, dict)

		if (TorchCheckpointKey.EPOCH.value in checkpoint_dict_form):
			self.epoch_index = checkpoint_dict_form[TorchCheckpointKey.EPOCH.value]
		
		if (TorchCheckpointKey.MODEL.value in checkpoint_dict_form):
			self.model_state_dict = checkpoint_dict_form[TorchCheckpointKey.MODEL.value]
		
		if (TorchCheckpointKey.OPTIMIZER.value in checkpoint_dict_form):
			self.optimizer_state_dict = checkpoint_dict_form[TorchCheckpointKey.OPTIMIZER.value]
		
		if (TorchCheckpointKey.CUSTOM.value in checkpoint_dict_form):
			self.custom_dict = checkpoint_dict_form[TorchCheckpointKey.CUSTOM.value]


class ResnetBlock(torch.nn.Module):
	"""
	A regular block for resnet (with no bottleneck).

	References:
		https://arxiv.org/abs/1512.03385
	"""

	def __init__(self, num_input_channels, downsample_rate, num_output_channels, use_bn=True):
		# Check arguments
		num_input_channels = int(num_input_channels)
		assert (num_input_channels >= 1)
		self._num_input_channels = num_input_channels

		downsample_rate = int(downsample_rate)
		assert (downsample_rate >= 1)
		self._downsample_rate = downsample_rate

		num_output_channels = int(num_output_channels)
		assert (num_output_channels >= 1)
		self._num_output_channels = num_output_channels

		use_bn = bool(use_bn)
		self._use_bn = use_bn

		# Call superclass constructor
		super(ResnetBlock, self).__init__()

		# Layer 1
		if use_bn:
			self.conv1 = torch.nn.Conv2d(num_input_channels, num_output_channels, 3, stride=downsample_rate, padding=1, bias=False)
			self.bn1 = torch.nn.BatchNorm2d(num_output_channels)
		else:
			self.conv1 = torch.nn.Conv2d(num_input_channels, num_output_channels, 3, stride=downsample_rate, padding=1, bias=True)
		
		self.relu1 = torch.nn.ReLU(inplace=False)

		# Layer 2
		if use_bn:
			self.conv2 = torch.nn.Conv2d(num_output_channels, num_output_channels, 3, stride=1, padding=1, bias=False)
			self.bn2 = torch.nn.BatchNorm2d(num_output_channels)
		else:
			self.conv2 = torch.nn.Conv2d(num_output_channels, num_output_channels, 3, stride=1, padding=1, bias=True)
		
		# Projection Layer
		if self.has_projection_layer:
			if use_bn:
				self.conv_proj = torch.nn.Conv2d(num_input_channels, num_output_channels, 1, stride=downsample_rate, padding=0, bias=False)
				self.bn_proj = torch.nn.BatchNorm2d(num_output_channels)
			else:
				self.conv_proj = torch.nn.Conv2d(num_input_channels, num_output_channels, 1, stride=downsample_rate, padding=0, bias=True)
		
		self.relu2 = torch.nn.ReLU(inplace=False)
	

	@property
	def num_input_channels(self):
		return self._num_input_channels
	

	@property
	def downsample_rate(self):
		return self._downsample_rate
	

	@property
	def num_output_channels(self):
		return self._num_output_channels
	

	@property
	def use_bn(self):
		return self._use_bn
	

	@property
	def has_projection_layer(self):
		return ((self.downsample_rate > 1) or (self.num_input_channels != self.num_output_channels))
	

	def reset_parameters(self):
		self.conv1.reset_parameters()
		if self.use_bn:
			self.bn1.reset_parameters()
		
		self.conv2.reset_parameters()
		if self.use_bn:
			self.bn2.reset_parameters()
		
		if self.has_projection_layer:
			self.conv_proj.reset_parameters()
			if self.use_bn:
				self.bn_proj.reset_parameters()
	

	def forward(self, input_variable):
		if self.use_bn:
			layer1_output_variable = self.relu1(self.bn1(self.conv1(input_variable)))
		else:
			layer1_output_variable = self.relu1(self.conv1(input_variable))
		
		if self.use_bn:
			conv2_output_variable = self.bn2(self.conv2(layer1_output_variable))
		else:
			conv2_output_variable = self.conv2(layer1_output_variable)
		
		if self.has_projection_layer:
			if self.use_bn:
				residual_variable = self.bn_proj(self.conv_proj(input_variable))
			else:
				residual_variable = self.conv_proj(input_variable)
		else:
			residual_variable = input_variable
		
		layer2_output_variable = self.relu2(conv2_output_variable + residual_variable)

		return layer2_output_variable


class ResnetBottleneckBlock(torch.nn.Module):
	"""
	A bottleneck block for resnet (with no bottleneck).

	References:
		https://arxiv.org/abs/1512.03385
	"""

	def __init__(self, num_input_channels, downsample_rate, num_bottleneck_channels, num_output_channels, use_bn=True):
		# Check arguments
		num_input_channels = int(num_input_channels)
		assert (num_input_channels >= 1)
		self._num_input_channels = num_input_channels

		downsample_rate = int(downsample_rate)
		assert (downsample_rate >= 1)
		self._downsample_rate = downsample_rate

		num_bottleneck_channels = int(num_bottleneck_channels)
		assert (num_bottleneck_channels >= 1)
		assert (num_bottleneck_channels <= num_input_channels)
		self._num_bottleneck_channels = num_bottleneck_channels

		num_output_channels = int(num_output_channels)
		assert (num_output_channels >= num_bottleneck_channels)
		self._num_output_channels = num_output_channels

		use_bn = bool(use_bn)
		self._use_bn = use_bn

		# Call superclass constructor
		super(ResnetBottleneckBlock, self).__init__()

		# Layer 1
		if use_bn:
			self.conv1 = torch.nn.Conv2d(num_input_channels, num_bottleneck_channels, 1, stride=1, padding=0, bias=False)
			self.bn1 = torch.nn.BatchNorm2d(num_bottleneck_channels)
		else:
			self.conv1 = torch.nn.Conv2d(num_input_channels, num_bottleneck_channels, 1, stride=1, padding=0, bias=True)
		
		self.relu1 = torch.nn.ReLU(inplace=False)

		# Layer 2
		if use_bn:
			self.conv2 = torch.nn.Conv2d(num_bottleneck_channels, num_bottleneck_channels, 3, stride=downsample_rate, padding=1, bias=False)
			self.bn2 = torch.nn.BatchNorm2d(num_bottleneck_channels)
		else:
			self.conv2 = torch.nn.Conv2d(num_bottleneck_channels, num_bottleneck_channels, 3, stride=downsample_rate, padding=1, bias=True)
		
		self.relu2 = torch.nn.ReLU(inplace=False)

		# Layer 3
		if use_bn:
			self.conv3 = torch.nn.Conv2d(num_bottleneck_channels, num_output_channels, 1, stride=1, padding=0, bias=False)
			self.bn3 = torch.nn.BatchNorm2d(num_output_channels)
		else:
			self.conv3 = torch.nn.Conv2d(num_bottleneck_channels, num_output_channels, 1, stride=1, padding=0, bias=True)
		
		# Projection Layer
		if self.has_projection_layer:
			if use_bn:
				self.conv_proj = torch.nn.Conv2d(num_input_channels, num_output_channels, 1, stride=downsample_rate, padding=0, bias=False)
				self.bn_proj = torch.nn.BatchNorm2d(num_output_channels)
			else:
				self.conv_proj = torch.nn.Conv2d(num_input_channels, num_output_channels, 1, stride=downsample_rate, padding=0, bias=True)
		
		self.relu3 = torch.nn.ReLU(inplace=False)
	

	@property
	def num_input_channels(self):
		return self._num_input_channels
	

	@property
	def downsample_rate(self):
		return self._downsample_rate
	

	@property
	def num_bottleneck_channels(self):
		return self._num_bottleneck_channels
	

	@property
	def num_output_channels(self):
		return self._num_output_channels
	

	@property
	def use_bn(self):
		return self._use_bn
	

	@property
	def has_projection_layer(self):
		return ((self.downsample_rate > 1) or (self.num_input_channels != self.num_output_channels))
	

	def reset_parameters(self):
		self.conv1.reset_parameters()
		if self.use_bn:
			self.bn1.reset_parameters()
		
		self.conv2.reset_parameters()
		if self.use_bn:
			self.bn2.reset_parameters()
		
		self.conv3.reset_parameters()
		if self.use_bn:
			self.bn3.reset_parameters()
		
		if self.has_projection_layer:
			self.conv_proj.reset_parameters()
			if self.use_bn:
				self.bn_proj.reset_parameters()
	

	def forward(self, input_variable):
		if self.use_bn:
			layer1_output_variable = self.relu1(self.bn1(self.conv1(input_variable)))
		else:
			layer1_output_variable = self.relu1(self.conv1(input_variable))
		
		if self.use_bn:
			layer2_output_variable = self.relu2(self.bn2(self.conv2(layer1_output_variable)))
		else:
			layer2_output_variable = self.relu2(self.conv2(layer1_output_variable))
		
		if self.use_bn:
			conv3_output_variable = self.bn3(self.conv3(layer2_output_variable))
		else:
			conv3_output_variable = self.conv3(layer2_output_variable)
		
		if self.has_projection_layer:
			if self.use_bn:
				residual_variable = self.bn_proj(self.conv_proj(input_variable))
			else:
				residual_variable = self.conv_proj(input_variable)
		else:
			residual_variable = input_variable
		
		layer3_output_variable = self.relu3(conv3_output_variable + residual_variable)

		return layer3_output_variable


class ResnetUnit(torch.nn.Sequential):
	"""
	A unit of resnet consisting of multiple regular (non-bottleneck) blocks.
	
	This corresponds to the conv2_x, conv3_x, conv4_x, and conv5_x structures in Table 1 of reference.
	I coined the term "unit" because I think "layer" (as used in the reference) is not an appropriate name for this structure.

	References:
		https://arxiv.org/abs/1512.03385
	"""

	def __init__(self, num_blocks, num_input_channels, downsample_rate, num_output_channels, use_bn=True):
		# Check arguments. Arguments to ResnetBlock are checked within that class and not checked here.
		num_blocks = int(num_blocks)
		assert (num_blocks >= 1)
		self._num_blocks = num_blocks

		# First block with downsampling
		attr_block_tuple_list = []
		attr_block_tuple_list.append((
			"block1", ResnetBlock(num_input_channels, downsample_rate, num_output_channels, use_bn)
		))

		# Other blocks without downsampling
		for block_index in range(1, num_blocks):
			attr_block_tuple_list.append((
				("block%d" % (block_index + 1, )), ResnetBlock(num_output_channels, 1, num_output_channels, use_bn)
			))

		# Call superclass constructor
		super(ResnetUnit, self).__init__(OrderedDict(attr_block_tuple_list))
	

	@property
	def num_input_channels(self):
		return self.block1.num_input_channels
	

	@property
	def downsample_rate(self):
		return self.block1.downsample_rate
	

	@property
	def num_output_channels(self):
		return getattr(self, ("block%d" % (len(self), ))).num_output_channels
	

	def reset_parameters(self):
		for submodule in self._modules.values():
			submodule.reset_parameters()


class ResnetBottleneckUnit(torch.nn.Sequential):
	"""
	A unit of resnet consisting of multiple bottleneck blocks.
	
	This corresponds to the conv2_x, conv3_x, conv4_x, and conv5_x structures in Table 1 of reference.
	I coined the term "unit" because I think "layer" (as used in the reference) is not an appropriate name for this structure.

	References:
		https://arxiv.org/abs/1512.03385
	"""

	def __init__(self, num_blocks, num_input_channels, downsample_rate, num_bottleneck_channels, num_output_channels, use_bn=True):
		# Check arguments. Arguments to ResnetBottleneckBlock are checked within that class and not checked here.
		num_blocks = int(num_blocks)
		assert (num_blocks >= 1)
		self._num_blocks = num_blocks

		# First block with downsampling
		attr_block_tuple_list = []
		attr_block_tuple_list.append((
			"block1", ResnetBottleneckBlock(num_input_channels, downsample_rate, num_bottleneck_channels, num_output_channels, use_bn)
		))

		# Other blocks without downsampling
		for block_index in range(1, num_blocks):
			attr_block_tuple_list.append((
				("block%d" % (block_index + 1, )), ResnetBottleneckBlock(num_output_channels, 1, num_bottleneck_channels, num_output_channels, use_bn)
			))

		# Call superclass constructor
		super(ResnetBottleneckUnit, self).__init__(OrderedDict(attr_block_tuple_list))
	

	@property
	def num_input_channels(self):
		return self.block1.num_input_channels
	

	@property
	def downsample_rate(self):
		return self.block1.downsample_rate
	

	@property
	def num_bottleneck_channels(self):
		return self.block1.num_bottleneck_channels
	

	@property
	def num_output_channels(self):
		return getattr(self, ("block%d" % (len(self), ))).num_output_channels
	

	def reset_parameters(self):
		for submodule in self._modules.values():
			submodule.reset_parameters()