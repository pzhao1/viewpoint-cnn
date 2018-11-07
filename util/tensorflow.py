from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def add_tensor_summaries(tensor_to_summarize, name_prefix):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	mean_tensor = tf.reduce_mean(tensor_to_summarize)
	standard_deviation_tensor = tf.sqrt(tf.reduce_mean(tf.square(tensor_to_summarize - mean_tensor)))

	tf.summary.scalar(name_prefix+"_mean", mean_tensor)
	tf.summary.scalar(name_prefix+"_stddev", standard_deviation_tensor)
	tf.summary.scalar(name_prefix+"_max", tf.reduce_max(tensor_to_summarize))
	tf.summary.scalar(name_prefix+"_min", tf.reduce_min(tensor_to_summarize))
	tf.summary.histogram(name_prefix+"_histogram", tensor_to_summarize)


def box_xywh_to_tlbr(box_xywh_tensor):
	assert isinstance(box_xywh_tensor, tf.Tensor)
	assert (box_xywh_tensor.dtype == tf.float32)
	assert (box_xywh_tensor.shape.ndims is not None)
	assert (box_xywh_tensor.shape.ndims > 0)
	assert (box_xywh_tensor.shape[-1].value == 4)

	with tf.name_scope("box_xywh_to_tlbr"):
		box_center_x_tensor = box_xywh_tensor[..., 0]
		box_center_y_tensor = box_xywh_tensor[..., 1]
		box_width_tensor = box_xywh_tensor[..., 2]
		box_height_tensor = box_xywh_tensor[..., 3]

		box_top_tensor = box_center_y_tensor - (box_height_tensor / 2.0)
		box_left_tensor = box_center_x_tensor - (box_width_tensor / 2.0)
		box_bottom_tensor = box_center_y_tensor + (box_height_tensor / 2.0)
		box_right_tensor = box_center_x_tensor + (box_width_tensor / 2.0)

		box_tlbr_tensor = tf.stack([box_top_tensor, box_left_tensor, box_bottom_tensor, box_right_tensor], axis=-1)
	
	return box_tlbr_tensor


def box_tlbr_to_xywh(box_tlbr_tensor):
	assert isinstance(box_tlbr_tensor, tf.Tensor)
	assert (box_tlbr_tensor.dtype == tf.float32)
	assert (box_tlbr_tensor.shape.ndims is not None)
	assert (box_tlbr_tensor.shape.ndims > 0)
	assert (box_tlbr_tensor.shape[-1].value == 4)

	with tf.name_scope("box_tlbr_to_xywh"):
		box_top_tensor = box_tlbr_tensor[..., 0]
		box_left_tensor = box_tlbr_tensor[..., 1]
		box_bottom_tensor = box_tlbr_tensor[..., 2]
		box_right_tensor = box_tlbr_tensor[..., 3]

		box_center_x_tensor = (box_left_tensor + box_right_tensor) / 2.0
		box_center_y_tensor = (box_top_tensor + box_bottom_tensor) / 2.0
		box_width_tensor = box_right_tensor - box_left_tensor
		box_height_tensor = box_bottom_tensor - box_top_tensor

		box_xywh_tensor = tf.stack([box_center_x_tensor, box_center_y_tensor, box_width_tensor, box_height_tensor], axis=-1)
	
	return box_xywh_tensor


def box_groups_iou(group1_box_tlbr_tensor, group2_box_tlbr_tensor):
	assert isinstance(group1_box_tlbr_tensor, tf.Tensor)
	assert (group1_box_tlbr_tensor.dtype == tf.float32)
	group1_box_tlbr_tensor.shape.assert_has_rank(2)
	assert (group1_box_tlbr_tensor.shape[1].value == 4)

	assert isinstance(group2_box_tlbr_tensor, tf.Tensor)
	assert (group2_box_tlbr_tensor.dtype == tf.float32)
	group2_box_tlbr_tensor.shape.assert_has_rank(2)
	assert (group2_box_tlbr_tensor.shape[1].value == 4)

	with tf.name_scope("box_groups_iou"):
		intersection_top_tensor = tf.maximum(tf.reshape(group1_box_tlbr_tensor[:, 0], [-1, 1]), tf.reshape(group2_box_tlbr_tensor[:, 0], [1, -1]))
		intersection_left_tensor = tf.maximum(tf.reshape(group1_box_tlbr_tensor[:, 1], [-1, 1]), tf.reshape(group2_box_tlbr_tensor[:, 1], [1, -1]))
		intersection_bottom_tensor = tf.minimum(tf.reshape(group1_box_tlbr_tensor[:, 2], [-1, 1]), tf.reshape(group2_box_tlbr_tensor[:, 2], [1, -1]))
		intersection_right_tensor = tf.minimum(tf.reshape(group1_box_tlbr_tensor[:, 3], [-1, 1]), tf.reshape(group2_box_tlbr_tensor[:, 3], [1, -1]))

		intersection_area_tensor = (
			tf.cast((intersection_left_tensor < intersection_right_tensor), tf.float32) * 
			tf.cast((intersection_top_tensor < intersection_bottom_tensor), tf.float32) * 
			(intersection_right_tensor - intersection_left_tensor) * (intersection_bottom_tensor - intersection_top_tensor)
		)

		group1_box_area_tensor = (
			(group1_box_tlbr_tensor[:, 2] - group1_box_tlbr_tensor[:, 0]) * 
			(group1_box_tlbr_tensor[:, 3] - group1_box_tlbr_tensor[:, 1])
		)
		group2_box_area_tensor = (
			(group2_box_tlbr_tensor[:, 2] - group2_box_tlbr_tensor[:, 0]) * 
			(group2_box_tlbr_tensor[:, 3] - group2_box_tlbr_tensor[:, 1])
		)
		iou_tensor = tf.truediv(
			intersection_area_tensor, 
			(tf.reshape(group1_box_area_tensor, [-1, 1]) + tf.reshape(group2_box_area_tensor, [1, -1]) - intersection_area_tensor)
		)
	
	return iou_tensor


def box_clip_to_boundaries(box_tlbr_tensor, min_x_tensor, max_x_tensor, min_y_tensor, max_y_tensor):
	assert isinstance(box_tlbr_tensor, tf.Tensor)
	assert (box_tlbr_tensor.dtype == tf.float32)
	assert (box_tlbr_tensor.shape.ndims is not None)
	assert (box_tlbr_tensor.shape.ndims > 0)
	assert (box_tlbr_tensor.shape[-1].value == 4)

	with tf.name_scope("box_clip_to_boundaries"):
		min_x_tensor = tf.cast(min_x_tensor, tf.float32)
		min_x_tensor.shape.assert_has_rank(0)

		max_x_tensor = tf.cast(max_x_tensor, tf.float32)
		max_x_tensor.shape.assert_has_rank(0)

		min_y_tensor = tf.cast(min_y_tensor, tf.float32)
		min_y_tensor.shape.assert_has_rank(0)

		max_y_tensor = tf.cast(max_y_tensor, tf.float32)
		max_y_tensor.shape.assert_has_rank(0)

		validate_assertion_list = [
			tf.assert_less(min_x_tensor, max_x_tensor), 
			tf.assert_less(min_y_tensor, max_y_tensor)
		]
		with tf.control_dependencies(validate_assertion_list):
			clipped_top_tensor = tf.clip_by_value(box_tlbr_tensor[..., 0], min_y_tensor, max_y_tensor)
			clipped_left_tensor = tf.clip_by_value(box_tlbr_tensor[..., 1], min_x_tensor, max_x_tensor)
			clipped_bottom_tensor = tf.clip_by_value(box_tlbr_tensor[..., 2], min_y_tensor, max_y_tensor)
			clipped_right_tensor = tf.clip_by_value(box_tlbr_tensor[..., 3], min_x_tensor, max_x_tensor)

			clipped_box_tlbr_tensor = tf.stack(
				[clipped_top_tensor, clipped_left_tensor, clipped_bottom_tensor, clipped_right_tensor], 
				axis=-1
			)
	
	return clipped_box_tlbr_tensor


def box_tlbr_normalize(box_tlbr_tensor, image_width_tensor, image_height_tensor):
	assert isinstance(box_tlbr_tensor, tf.Tensor)
	assert (box_tlbr_tensor.dtype == tf.float32)
	assert (box_tlbr_tensor.shape.ndims is not None)
	assert (box_tlbr_tensor.shape.ndims > 0)
	assert (box_tlbr_tensor.shape[-1].value == 4)

	with tf.name_scope("box_tlbr_normalize"):
		image_width_tensor = tf.cast(image_width_tensor, tf.float32)
		image_width_tensor.shape.assert_has_rank(0)

		image_height_tensor = tf.cast(image_height_tensor, tf.float32)
		image_height_tensor.shape.assert_has_rank(0)

		validate_assertion_list = [
			tf.assert_greater(image_width_tensor, 1.0), 
			tf.assert_greater(image_height_tensor, 1.0)
		]
		with tf.control_dependencies(validate_assertion_list):
			image_width_minus_one_tensor = (image_width_tensor - 1.0)
			image_height_minus_one_tensor = (image_height_tensor - 1.0)

			normalized_top_tensor = tf.truediv(box_tlbr_tensor[..., 0], image_height_minus_one_tensor)
			normalized_left_tensor = tf.truediv(box_tlbr_tensor[..., 1], image_width_minus_one_tensor)
			normalized_bottom_tensor = tf.truediv(box_tlbr_tensor[..., 2], image_height_minus_one_tensor)
			normalized_right_tensor = tf.truediv(box_tlbr_tensor[..., 3], image_width_minus_one_tensor)

			normalized_box_tlbr_tensor = tf.stack(
				[normalized_top_tensor, normalized_left_tensor, normalized_bottom_tensor, normalized_right_tensor], 
				axis=-1
			)
	
	return normalized_box_tlbr_tensor


def resnet_block(input_tensor, is_training_tensor, downsample_rate, regularizer_scale, use_bn, num_output_channels):
	"""
	A regular block for resnet (with no bottleneck).

	References:
		https://arxiv.org/abs/1512.03385
	"""
	# Check arguments
	# Only want float32 for input, because float16 is not accurate enough, and float64 is to computationally intensive.
	assert isinstance(input_tensor, tf.Tensor)
	assert (input_tensor.dtype == tf.float32)
	input_tensor.shape.assert_has_rank(4)
	assert (input_tensor.shape[3].value is not None)
	num_input_channels = input_tensor.shape[3].value

	assert isinstance(is_training_tensor, tf.Tensor)
	assert (is_training_tensor.dtype == tf.bool)

	downsample_rate = int(downsample_rate)
	assert (downsample_rate >= 1)

	regularizer_scale = float(regularizer_scale)
	assert (regularizer_scale >= 0.0)

	assert isinstance(use_bn, bool)

	num_output_channels = int(num_output_channels)
	assert (num_output_channels >= 1)

	# First layer (weight + normalization + relu)
	with tf.variable_scope("layer1"):
		if use_bn:
			conv1_tensor = tf.layers.conv2d(
				input_tensor, num_output_channels, 3, strides=downsample_rate, padding="SAME", use_bias=False, 
				kernel_initializer=tf.contrib.layers.xavier_initializer(), 
				kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale)
			)
			bn1_tensor = tf.layers.batch_normalization(
				conv1_tensor, training=is_training_tensor, trainable=True
			)
			relu1_tensor = tf.nn.relu(bn1_tensor)
		else:
			conv1_tensor = tf.layers.conv2d(
				input_tensor, num_output_channels, 3, strides=downsample_rate, padding="SAME", use_bias=True, 
				kernel_initializer=tf.contrib.layers.xavier_initializer(), 
				bias_initializer=tf.contrib.layers.xavier_initializer(), 
				kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale), 
				bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale)
			)
			relu1_tensor = tf.nn.relu(conv1_tensor)
		
		layer1_output_tensor = relu1_tensor
	
	# Second layer (weight + normalization)
	with tf.variable_scope("layer2"):
		if use_bn:
			conv2_tensor = tf.layers.conv2d(
				layer1_output_tensor, num_output_channels, 3, padding="SAME", use_bias=False, 
				kernel_initializer=tf.contrib.layers.xavier_initializer(), 
				kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale)
			)
			bn2_tensor = tf.layers.batch_normalization(
				conv2_tensor, training=is_training_tensor, trainable=True
			)
			layer2_output_tensor = bn2_tensor
		else:
			conv2_tensor = tf.layers.conv2d(
				layer1_output_tensor, num_output_channels, 3, padding="SAME", use_bias=True, 
				kernel_initializer=tf.contrib.layers.xavier_initializer(), 
				bias_initializer=tf.contrib.layers.xavier_initializer(), 
				kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale), 
				bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale)
			)
			layer2_output_tensor = conv2_tensor
	
	# Add identity or projection transform (so a residual is learned)
	with tf.variable_scope("addition"):
		if ((num_input_channels == num_output_channels) and (downsample_rate == 1)):
			addition_tensor = layer2_output_tensor + input_tensor
		else:
			projected_input_tensor = tf.layers.conv2d(
				input_tensor, num_output_channels, 1, strides=downsample_rate, padding="SAME", use_bias=False, 
				kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=None
			)
			addition_tensor = layer2_output_tensor + projected_input_tensor
		
	block_output_tensor = tf.nn.relu(addition_tensor)
	return block_output_tensor


def resnet_bottleneck_block(input_tensor, is_training_tensor, num_bottleneck_channels, downsample_rate, 
		regularizer_scale, use_bn, num_output_channels):
	"""
	A bottlenecked block for resnet.

	References:
		https://arxiv.org/abs/1512.03385
	"""
	# Check arguments
	# Only want float32 for input, because float16 is not accurate enough, and float64 is to computationally intensive.
	assert isinstance(input_tensor, tf.Tensor)
	assert (input_tensor.dtype == tf.float32)
	input_tensor.shape.assert_has_rank(4)
	assert (input_tensor.shape[3].value is not None)
	num_input_channels = input_tensor.shape[3].value

	assert isinstance(is_training_tensor, tf.Tensor)
	assert (is_training_tensor.dtype == tf.bool)

	num_bottleneck_channels = int(num_bottleneck_channels)
	assert (num_bottleneck_channels >= 1)
	assert (num_bottleneck_channels <= num_input_channels)

	downsample_rate = int(downsample_rate)
	assert (downsample_rate >= 1)

	regularizer_scale = float(regularizer_scale)
	assert (regularizer_scale >= 0.0)

	assert isinstance(use_bn, bool)

	num_output_channels = int(num_output_channels)
	assert (num_output_channels >= num_bottleneck_channels)

	# First layer (weight + normalization + relu)
	with tf.variable_scope("layer1"):
		if use_bn:
			conv1_tensor = tf.layers.conv2d(
				input_tensor, num_bottleneck_channels, 1, padding="SAME", use_bias=False, 
				kernel_initializer=tf.contrib.layers.xavier_initializer(), 
				kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale)
			)
			bn1_tensor = tf.layers.batch_normalization(
				conv1_tensor, training=is_training_tensor, trainable=True
			)
			relu1_tensor = tf.nn.relu(bn1_tensor)
		else:
			conv1_tensor = tf.layers.conv2d(
				input_tensor, num_bottleneck_channels, 1, padding="SAME", use_bias=True, 
				kernel_initializer=tf.contrib.layers.xavier_initializer(), 
				bias_initializer=tf.contrib.layers.xavier_initializer(), 
				kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale), 
				bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale)
			)
			relu1_tensor = tf.nn.relu(conv1_tensor)
		
		layer1_output_tensor = relu1_tensor
	
	# Second layer (weight + normalization + relu)
	with tf.variable_scope("layer2"):
		if use_bn:
			conv2_tensor = tf.layers.conv2d(
				layer1_output_tensor, num_bottleneck_channels, 3, strides=downsample_rate, padding="SAME", use_bias=False,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),  
				kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale)
			)
			bn2_tensor = tf.layers.batch_normalization(
				conv2_tensor, training=is_training_tensor, trainable=True
			)
			relu2_tensor = tf.nn.relu(bn2_tensor)
		else:
			conv2_tensor = tf.layers.conv2d(
				layer1_output_tensor, num_bottleneck_channels, 3, strides=downsample_rate, padding="SAME", use_bias=True, 
				kernel_initializer=tf.contrib.layers.xavier_initializer(), 
				bias_initializer=tf.contrib.layers.xavier_initializer(), 
				kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale), 
				bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale)
			)
			relu2_tensor = tf.nn.relu(conv2_tensor)
		
		layer2_output_tensor = relu2_tensor
	
	# Third layer (weight + normalization)
	with tf.variable_scope("layer3"):
		if use_bn:
			conv3_tensor = tf.layers.conv2d(
				layer2_output_tensor, num_output_channels, 1, padding="SAME", use_bias=False, 
				kernel_initializer=tf.contrib.layers.xavier_initializer(), 
				kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale)
			)
			bn3_tensor = tf.layers.batch_normalization(
				conv3_tensor, training=is_training_tensor, trainable=True
			)
			layer3_output_tensor = bn3_tensor
		else:
			conv3_tensor = tf.layers.conv2d(
				layer2_output_tensor, num_output_channels, 1, padding="SAME", use_bias=True, 
				kernel_initializer=tf.contrib.layers.xavier_initializer(), 
				bias_initializer=tf.contrib.layers.xavier_initializer(), 
				kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale), 
				bias_regularizer=tf.contrib.layers.l2_regularizer(regularizer_scale)
			)
			layer3_output_tensor = conv3_tensor
	
	# Add identity or projection transform (so a residual is learned)
	with tf.variable_scope("addition"):
		if (num_input_channels == num_output_channels):
			addition_tensor = layer3_output_tensor + input_tensor
		else:
			projected_input_tensor = tf.layers.conv2d(
				input_tensor, num_output_channels, 1, strides=downsample_rate, padding="SAME", use_bias=False, 
				kernel_initializer=tf.contrib.layers.xavier_initializer(), kernel_regularizer=None
			)
			addition_tensor = layer3_output_tensor + projected_input_tensor
		
	block_output_tensor = tf.nn.relu(addition_tensor)
	return block_output_tensor


def resnet_unit(input_tensor, is_training_tensor, num_blocks, downsample_rate, regularizer_scale, use_bn, num_output_channels):
	"""
	A unit of resnet consisting of multiple regular (non-bottleneck) blocks.
	
	This corresponds to the conv2_x, conv3_x, conv4_x, and conv5_x structures in Table 1 of reference.
	I coined the term "unit" because I think "layer" (as used in the reference) is not an appropriate name for this structure.

	Reference:
		https://arxiv.org/abs/1512.03385
	"""
	# Check arguments. Arguments to resnet_block are checked within that function and not checked here.
	num_blocks = int(num_blocks)
	assert (num_blocks >= 1)

	# Construct first block with downsampling.
	with tf.variable_scope("block1"):
		block_output_tensor = resnet_block(
			input_tensor, is_training_tensor, downsample_rate, regularizer_scale, use_bn, num_output_channels
		)
	
	# Construct other blocks without downsampling.
	for block_index in range(1, num_blocks):
		with tf.variable_scope("block%d" % (block_index + 1, )):
			block_output_tensor = resnet_block(
				block_output_tensor, is_training_tensor, 1, regularizer_scale, use_bn, num_output_channels
			)
	
	return block_output_tensor


def resnet_bottleneck_unit(input_tensor, is_training_tensor, num_blocks, num_bottleneck_channels, downsample_rate, 
		regularizer_scale, use_bn, num_output_channels):
	"""
	A unit of resnet consisting of multiple bottleneck blocks.
	
	This corresponds to the conv2_x, conv3_x, conv4_x, and conv5_x structures in Table 1 of reference.
	I coined the term "unit" because I think "layer" (as used in the reference) is not an appropriate name for this structure.

	Reference:
		https://arxiv.org/abs/1512.03385
	"""
	# Check arguments. Arguments to resnet_bottleneck_block are checked within that function and not checked here.
	num_blocks = int(num_blocks)
	assert (num_blocks >= 1)

	# Construct first block with downsampling.
	with tf.variable_scope("block1"):
		block_output_tensor = resnet_bottleneck_block(
			input_tensor, is_training_tensor, num_bottleneck_channels, downsample_rate, regularizer_scale, use_bn, num_output_channels
		)
	
	# Construct the blocks
	for block_index in range(1, num_blocks):
		with tf.variable_scope("block%d" % (block_index + 1, )):
			block_output_tensor = resnet_bottleneck_block(
				block_output_tensor, is_training_tensor, num_bottleneck_channels, 1, regularizer_scale, use_bn, num_output_channels
			)
	
	return block_output_tensor