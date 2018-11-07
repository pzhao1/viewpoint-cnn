from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, "../.."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import util.statistics


def sample_truncated_normal_test1():
	left_clip = -np.pi / 2.0
	right_clip = np.pi / 2.0
	mean = np.deg2rad(20.0)
	sigma = np.deg2rad(40.0)
	
	sample_array = util.statistics.sample_truncated_normal(left_clip, right_clip, mean, sigma, (300000))

	assert (not np.any(sample_array < left_clip, axis=None))
	assert (not np.any(sample_array > right_clip, axis=None))
	
	matplotlib.pyplot.hist(sample_array, 50)
	matplotlib.pyplot.show()