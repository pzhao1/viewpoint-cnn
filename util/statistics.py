from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.interpolate
import scipy.special
import scipy.stats


def softmax(logit_array, axis_index, minimum_value=0.0):
	assert isinstance(logit_array, np.ndarray)
	assert (logit_array.ndim >= 1)
	assert np.issubdtype(logit_array.dtype, np.floating)

	axis_index = int(axis_index)
	assert (axis_index >= 0)
	assert (axis_index < logit_array.ndim)

	minimum_value = float(minimum_value)
	assert (minimum_value >= 0.0)
	assert (minimum_value <= 1e-4)

	max_entry = np.amax(logit_array, axis=axis_index, keepdims=True)
	normalized_logit_array = logit_array - max_entry
	
	exp_array = np.exp(normalized_logit_array)
	normalized_exp_array = exp_array / np.sum(exp_array, axis=axis_index, keepdims=True)

	clipped_exp_array = np.maximum(normalized_exp_array, minimum_value)
	clipped_exp_array = clipped_exp_array / np.sum(clipped_exp_array, axis=axis_index, keepdims=True)

	return clipped_exp_array


def sample_truncated_normal(left_clip, right_clip, mean, sigma, size):
	left_clip = float(left_clip)
	right_clip = float(right_clip)
	assert (right_clip > left_clip)
	mean = float(mean)
	sigma = float(sigma)
	assert (sigma > 0.0)

	standard_left_clip = (left_clip - mean) / sigma
	standard_right_clip = (right_clip - mean) / sigma
	standard_sample_array = scipy.stats.truncnorm.rvs(standard_left_clip, standard_right_clip, size=size)

	sample_array = (standard_sample_array * sigma) + mean
	return sample_array


# def _test_sample_truncated_normal():
# 	left_clip = -np.pi / 6.0
# 	right_clip = np.pi / 2.0
# 	mean = np.deg2rad(20.0)
# 	sigma = np.deg2rad(20.0)
# 	sample_vector = sample_truncated_normal(left_clip, right_clip, mean, sigma, (300000))

# 	assert (not np.any(sample_vector < left_clip, axis=None))
# 	assert (not np.any(sample_vector > right_clip, axis=None))

# 	import matplotlib.pyplot
# 	matplotlib.pyplot.hist(sample_vector, 50)
# 	matplotlib.pyplot.show()


def compute_von_mises_vector(mu, kappa, num_points):
	mu = float(mu)
	kappa = float(kappa)
	assert (kappa >= 0.0)
	num_points = int(num_points)
	assert (num_points > 0)

	angle_rad_vector = np.linspace(0.0, 2.0 * np.pi, num=num_points, endpoint=False, dtype=np.float64)
	exponent_vector = kappa * np.cos(angle_rad_vector - mu)
	raw_prob_vector = np.exp(exponent_vector - np.amax(exponent_vector, axis=None))
	prob_vector = raw_prob_vector / np.sum(raw_prob_vector, axis=None)

	return prob_vector


def compute_von_mises_mixture_vector(mix_vector, mu_vector, kappa_vector, num_points):
	assert isinstance(mix_vector, np.ndarray)
	assert np.issubdtype(mix_vector.dtype, np.floating)
	assert (mix_vector.ndim == 1)
	assert np.all(mix_vector >= 0.0, axis=None)
	assert (np.sum(mix_vector, axis=None) > 0.0)
	num_components = mix_vector.size
	
	assert isinstance(mu_vector, np.ndarray)
	assert np.issubdtype(mu_vector.dtype, np.floating)
	assert (mu_vector.ndim == 1)
	assert (mu_vector.size == num_components)

	assert isinstance(kappa_vector, np.ndarray)
	assert np.issubdtype(kappa_vector.dtype, np.floating)
	assert (kappa_vector.ndim == 1)
	assert (kappa_vector.size == num_components)
	assert np.all(kappa_vector >= 0.0, axis=None)
	
	num_points = int(num_points)
	assert (num_points > 0)

	cumulative_prob_vector = np.zeros((num_points, ), dtype=np.float64)
	for component_index in range(num_components):
		component_prob_vector = compute_von_mises_vector(mu_vector[component_index], kappa_vector[component_index], num_points)
		cumulative_prob_vector = cumulative_prob_vector + (mix_vector[component_index] * component_prob_vector)
	
	prob_vector = cumulative_prob_vector / np.sum(cumulative_prob_vector, axis=None)
	return prob_vector


def von_mises_ml(prob_vector):
	assert isinstance(prob_vector, np.ndarray)
	assert (prob_vector.ndim == 1)
	assert np.issubdtype(prob_vector.dtype, np.floating)

	num_bins = prob_vector.shape[0]
	bin_angle_rad_vector = np.linspace(0.0, (2.0 * np.pi), num=num_bins, endpoint=False, dtype=np.float64)
	
	bin_angle_sin_vector = np.sin(bin_angle_rad_vector)
	weighted_bin_angle_sin_vector = prob_vector * bin_angle_sin_vector
	
	bin_angle_cos_vector = np.cos(bin_angle_rad_vector)
	weighted_bin_angle_cos_vector = prob_vector * bin_angle_cos_vector

	von_mises_mean_rad = np.arctan2(
		np.sum(weighted_bin_angle_sin_vector, axis=None, dtype=np.float64), 
		np.sum(weighted_bin_angle_cos_vector, axis=None, dtype=np.float64)
	)

	# Do not need concentration parameter now. Implement it if needed later.

	return von_mises_mean_rad


def _inverse_bessel_ratio(input_array):
	assert isinstance(input_array, np.ndarray)
	assert np.issubdtype(input_array.dtype, np.floating)
	assert (np.amin(input_array, axis=None) >= 0.0)
	assert (np.amax(input_array, axis=None) < 1.0)
	assert (not np.any(np.isnan(input_array), axis=None))

	function_self = _inverse_bessel_ratio
	if ((not hasattr(function_self, "lookup_array")) or (not hasattr(function_self, "lookup_min_input")) or (not hasattr(function_self, "lookup_max_input"))):
		lookup_min_output = 0.0
		lookup_max_output = 100.0
		lookup_num_samples = 100001

		output_sample_array = np.linspace(lookup_min_output, lookup_max_output, num=lookup_num_samples, endpoint=True, dtype=np.float64)
		input_sample_array = scipy.special.iv(1, output_sample_array)/scipy.special.iv(0, output_sample_array)
		input_sample_array = input_sample_array.astype(np.float64)

		lookup_min_input = np.amin(input_sample_array, axis=None)
		lookup_max_input = np.amax(input_sample_array, axis=None)
		lookup_input_array = np.linspace(lookup_min_input, lookup_max_input, num=lookup_num_samples, endpoint=True, dtype=np.float64)
		lookup_interpolator = scipy.interpolate.interp1d(
			input_sample_array, output_sample_array, kind="linear", axis=-1, copy=True, bounds_error=True, assume_sorted=False
		)
		lookup_output_array = lookup_interpolator(lookup_input_array)

		function_self.lookup_array = lookup_output_array
		function_self.lookup_min_input = lookup_min_input
		function_self.lookup_max_input = lookup_max_input

	do_lookup_mask = np.logical_and((input_array >= function_self.lookup_min_input), (input_array < function_self.lookup_max_input))
	
	lookup_granularity = float(function_self.lookup_max_input - function_self.lookup_min_input) / float(function_self.lookup_array.size - 1)
	lookup_index_array = np.floor(((input_array - function_self.lookup_min_input) * do_lookup_mask) / lookup_granularity)
	lookup_index_array = lookup_index_array.astype(np.int32)
	lookup_interp_ratio_array = np.mod(input_array, lookup_granularity) / lookup_granularity
	lookup_interp_low_array = function_self.lookup_array[lookup_index_array]
	lookup_interp_high_array = function_self.lookup_array[lookup_index_array+1]
	lookup_result_array = (lookup_interp_low_array * (1.0 - lookup_interp_ratio_array)) + (lookup_interp_high_array * lookup_interp_ratio_array)

	approximate_result_array = ((input_array * 2) - np.power(input_array, 3)) / (1 - np.power(input_array, 2))

	final_result_array = (do_lookup_mask * lookup_result_array) + (np.logical_not(do_lookup_mask) * approximate_result_array)
	return final_result_array


def _test_inverse_bessel_ratio():
	import datetime
	import matplotlib.pyplot

	kappa_vector = np.linspace(0, 50, num=361, endpoint=True, dtype=np.float64)
	bessel_ratio_vector = scipy.special.iv(1, kappa_vector) / scipy.special.iv(0, kappa_vector)

	start_time = datetime.datetime.now()
	#approximate_kappa_vector = ((bessel_ratio_vector * 2) - np.power(bessel_ratio_vector, 3)) / (1 - np.power(bessel_ratio_vector, 2))
	approximate_kappa_vector = _inverse_bessel_ratio(bessel_ratio_vector)
	end_time = datetime.datetime.now()

	mean_abs_error = np.mean(np.abs(approximate_kappa_vector - kappa_vector), axis=None)
	print("Mean abs error of approximation: %0.8f" % (mean_abs_error, ))
	print("Approximation took %0.8f seconds" % ((end_time - start_time).total_seconds(), ))
	
	matplotlib.pyplot.plot(bessel_ratio_vector, kappa_vector, 'k-', bessel_ratio_vector, approximate_kappa_vector, 'r-')
	matplotlib.pyplot.show()


def _von_mises_mixture_em_initialize(prob_vector, num_components, mix_prior):
	rho_vector = mix_prior * np.ones((num_components, ), dtype=np.float64)

	num_points = prob_vector.size
	max_prob_index = np.argmax(prob_vector, axis=None)
	max_prob_angle = (float(max_prob_index) * (2.0 * np.pi)) / float(num_points)
	mu_vector = np.linspace(max_prob_angle, max_prob_angle + (2.0 * np.pi), num=num_components, endpoint=False, dtype=np.float64)
	mu_vector = np.mod(mu_vector, (2.0 * np.pi))

	kappa_vector = np.ones((num_components, ), dtype=np.float64)

	return rho_vector, mu_vector, kappa_vector


def _von_mises_mixture_em_expectation(num_points, rho_vector, mu_vector, kappa_vector):
	num_components = rho_vector.size

	von_mises_matrix = np.zeros((num_points, num_components), dtype=np.float64)
	for component_index in range(num_components):
		von_mises_matrix[:, component_index] = compute_von_mises_vector(mu_vector[component_index], kappa_vector[component_index], num_points)
	
	expect_log_mix_vector = scipy.special.psi(rho_vector) - scipy.special.psi(np.sum(rho_vector, axis=None))
	mix_multiplier_vector = np.exp(expect_log_mix_vector - np.max(expect_log_mix_vector, axis=None))

	raw_expect_matrix = np.reshape(mix_multiplier_vector, (1, num_components)) * von_mises_matrix
	raw_expect_matrix = np.maximum(raw_expect_matrix, 1e-10)
	expect_matrix = raw_expect_matrix / np.sum(raw_expect_matrix, axis=1, keepdims=True)
	return expect_matrix


def _von_mises_mixture_em_maximization(prob_vector, expect_matrix, mix_prior, kappa_range, num_virtual_samples):
	(num_points, num_components) = expect_matrix.shape
	(min_kappa, max_kappa) = kappa_range

	scaled_expect_matrix = num_virtual_samples * (np.reshape(prob_vector, (num_points, 1)) * expect_matrix)

	data_angle_vector = np.linspace(0.0, 2.0 * np.pi, num=num_points, endpoint=False, dtype=np.float64)
	sum_sin_vector = np.sum(np.reshape(np.sin(data_angle_vector), (num_points, 1)) * scaled_expect_matrix, axis=0, keepdims=False)
	sum_cos_vector = np.sum(np.reshape(np.cos(data_angle_vector), (num_points, 1)) * scaled_expect_matrix, axis=0, keepdims=False)

	rho_vector = mix_prior + np.sum(scaled_expect_matrix, axis=0, keepdims=False)
	
	mu_vector = np.mod(np.arctan2(sum_sin_vector, sum_cos_vector), 2.0 * np.pi)
	
	rbar_vector = np.sqrt(np.power(sum_sin_vector, 2) + np.power(sum_cos_vector, 2)) / np.sum(scaled_expect_matrix, axis=0, keepdims=False)
	kappa_vector = _inverse_bessel_ratio(rbar_vector)
	kappa_vector = np.minimum(np.maximum(kappa_vector, min_kappa), max_kappa)

	return (rho_vector, mu_vector, kappa_vector)


def _von_mises_mixture_em_lower_bound(prob_vector, expect_matrix, rho_vector, mu_vector, kappa_vector, mix_prior, num_virtual_samples):
	(num_points, num_components) = expect_matrix.shape

	scaled_expect_matrix = num_virtual_samples * (np.reshape(prob_vector, (num_points, 1)) * expect_matrix)
	expect_log_mix_vector = scipy.special.psi(rho_vector) - scipy.special.psi(np.sum(rho_vector, axis=None))

	mix_term = (
		scipy.special.gammaln(num_components * mix_prior) - 
		(num_components * scipy.special.gammaln(mix_prior)) + 
		(mix_prior - 1.0) * np.sum(expect_log_mix_vector, axis=None)
	)

	expect_term = np.sum(scaled_expect_matrix * expect_log_mix_vector, axis=None)

	von_mises_matrix = np.zeros((num_points, num_components), dtype=np.float64)
	for component_index in range(num_components):
		von_mises_matrix[:, component_index] = compute_von_mises_vector(mu_vector[component_index], kappa_vector[component_index], num_points)
	data_term = np.sum(scaled_expect_matrix * np.log(von_mises_matrix), axis=None)

	mix_entropy_term = (-1.0) * (
		scipy.special.gammaln(np.sum(rho_vector, axis=None)) - 
		np.sum(scipy.special.gammaln(rho_vector), axis=None) +
		np.sum((rho_vector - 1.0) * expect_log_mix_vector, axis=None)
	)

	expect_entropy_term = (-1.0) * np.sum(scaled_expect_matrix * np.log(expect_matrix))

	lower_bound = mix_term + expect_term + data_term + mix_entropy_term + expect_entropy_term
	return lower_bound


def von_mises_mixture_em(prob_vector, num_components, mix_prior, kappa_range, num_virtual_samples, max_num_steps):
	assert isinstance(prob_vector, np.ndarray)
	assert (prob_vector.ndim == 1)
	assert np.issubdtype(prob_vector.dtype, np.floating)
	assert np.all(prob_vector >= 0.0, axis=None)
	prob_vector = prob_vector.astype(np.float64)
	num_points = prob_vector.size
	num_components = int(num_components)
	assert (num_components > 0)
	
	mix_prior = float(mix_prior)
	assert (mix_prior > 0.0)
	(min_kappa, max_kappa) = kappa_range
	min_kappa = float(min_kappa)
	max_kappa = float(max_kappa)
	assert (min_kappa >= 0.0)
	assert (max_kappa >= min_kappa)
	kappa_range = (min_kappa, max_kappa)
	
	num_virtual_samples = int(num_virtual_samples)
	assert (num_virtual_samples > 0)
	max_num_steps = int(max_num_steps)
	assert (max_num_steps > 0)
	# lower_bound_threshold = float(lower_bound_threshold)
	# assert (lower_bound_threshold >= 0.0)


	(rho_vector, mu_vector, kappa_vector) = _von_mises_mixture_em_initialize(prob_vector, num_components, mix_prior)

	for step_index in range(max_num_steps):
		expect_matrix = _von_mises_mixture_em_expectation(num_points, rho_vector, mu_vector, kappa_vector)
		lower_bound = _von_mises_mixture_em_lower_bound(prob_vector, expect_matrix, rho_vector, mu_vector, kappa_vector, mix_prior, num_virtual_samples)
		# print("Lower bound after step %03d E: %0.8f" %(step_index, lower_bound))

		(rho_vector, mu_vector, kappa_vector) = _von_mises_mixture_em_maximization(prob_vector, expect_matrix, mix_prior, kappa_range, num_virtual_samples)
		lower_bound = _von_mises_mixture_em_lower_bound(prob_vector, expect_matrix, rho_vector, mu_vector, kappa_vector, mix_prior, num_virtual_samples)
		# print("Lower bound after step %03d M: %0.8f" %(step_index, lower_bound))

		# if (step_index >= 1):
		# 	if (((lower_bound - previous_lower_bound) / num_virtual_samples) < lower_bound_threshold):
		# 		print ("EM stopped early at step %d, with a lower bound increment %0.16f" % (step_index, lower_bound - previous_lower_bound))
		# 		break
		# previous_lower_bound = lower_bound
	
	mix_vector = rho_vector / np.sum(rho_vector, axis=None)
	return (mix_vector, mu_vector, kappa_vector)


def _test_von_mises_mixture_em():
	import datetime

	num_points = 360
	
	# num_components = 3
	# gt_mix_vector = np.array([2/4, 2/4], dtype=np.float64)
	# gt_mu_vector = np.array([np.pi, 1.5*np.pi], dtype=np.float64)
	# gt_kappa_vector = np.array([4, 4], dtype=np.float64)

	num_components = 4
	gt_mix_vector = np.array([1/3, 1/3, 1/3], dtype=np.float64)
	gt_mu_vector = np.array([np.pi/2, np.pi, 3*np.pi/2], dtype=np.float64)
	gt_kappa_vector = np.array([2, 2, 2], dtype=np.float64)

	prob_vector = compute_von_mises_mixture_vector(gt_mix_vector, gt_mu_vector, gt_kappa_vector, num_points)

	_inverse_bessel_ratio(np.array([0.5], dtype=np.float64)) # Call this function once to build lookup table. Further calls will avoid this overhead.
	start_time = datetime.datetime.now()
	(est_mix_vector, est_mu_vector, est_kappa_vector) = von_mises_mixture_em(prob_vector, num_components, 0.1, (0.0, np.inf), 360, 200)
	end_time = datetime.datetime.now()
	print("\nApproximation took %0.8f seconds" % ((end_time - start_time).total_seconds(), ))

	print("\ngt  mix vector: ", gt_mix_vector)
	print("est mix vector: ", est_mix_vector)
	print("\ngt  mu vector: ", gt_mu_vector)
	print("est mu vector: ", est_mu_vector)
	print("\ngt kappa vector: ", gt_kappa_vector)
	print("est kappa vector: ", est_kappa_vector)