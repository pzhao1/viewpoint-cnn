"""
Render images for a ShapeNet model.

References:
	Joint embeddings of shapes and images via CNN image purification, Y. Li, H. Su, C. Qi et al.,  2015
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
import argparse

this_file_directory= os.path.dirname(os.path.abspath(__file__))
project_base_directory = os.path.abspath(os.path.join(this_file_directory, "../.."))
if (project_base_directory not in sys.path):
	sys.path.append(project_base_directory)

import global_variables as global_vars
import data.shapenet.meta_pb2 # If this is not found, bash ../../build_protobuf.sh
import render_config


def construct_argument_parser():
	argument_parser = argparse.ArgumentParser(
		prog="render_synset.py", 
		description="Render images for a ShapeNet synset."
	)
	argument_parser.add_argument(
		"shapenet_synset", 
		help="The ShapeNet synset to render images for"
	)
	return argument_parser


def prepare_synset_output_directory(shapenet_synset):
	# Check arguments
	shapenet_synset = str(shapenet_synset)

	# Construct synset output directory
	synset_output_dir = os.path.join(global_vars.SHAPENET_RENDER_DIR, shapenet_synset)
	if (os.path.isdir(synset_output_dir)):
		user_input = input("\nOutput directory %s/ already exists! Do you want to delete it? (y/n): " % (synset_output_dir)).lower()
		if (user_input == "y"):
			shutil.rmtree(synset_output_dir)
		else:
			exit()
	os.makedirs(synset_output_dir)

	# Construct an RSSynsetMeta object with an empty model_meta_list.
	# The model_meta_list will be populated by render_model.py.
	synset_meta_pbobj = data.shapenet.meta_pb2.RSSynsetMeta()
	synset_meta_header_pbobj = synset_meta_pbobj.header
	synset_meta_header_pbobj.shapenet_synset = shapenet_synset
	# Images are rendered from a number of random configurations.
	# The total number of these configurations are stored in the header.
	synset_meta_header_pbobj.num_configs = render_config.NUM_CONFIGS_PER_MODEL
	synset_meta_serialized_string = synset_meta_pbobj.SerializeToString()
	synset_meta_header_length = len(synset_meta_serialized_string)

	# Write RSSynsetMeta to file.
	# Later render_model.py will read the header of this file, and append to the end of this file.
	synset_meta_file_path = os.path.join(synset_output_dir, global_vars.SHAPENET_META_FILE_NAME)
	synset_meta_file = open(synset_meta_file_path, mode="wb")
	synset_meta_file.write(synset_meta_serialized_string)
	synset_meta_file.close()

	return (synset_output_dir, synset_meta_file_path, synset_meta_header_length)


def main():
	# Parse command line arguments
	argument_parser = construct_argument_parser()
	argument_namespace = argument_parser.parse_args()
	shapenet_synset = argument_namespace.shapenet_synset
	shapenet_synset_dir = os.path.join(global_vars.SHAPENET_DATA_DIR, shapenet_synset)
	assert os.path.isdir(shapenet_synset_dir)

	# Setup synset output directory
	(synset_output_dir, synset_meta_file_path, synset_meta_header_length) = prepare_synset_output_directory(shapenet_synset)

	# Main loop
	shapenet_synset_dir_entry_list = os.listdir(shapenet_synset_dir)
	model_index = 0
	for shapenet_synset_dir_entry in shapenet_synset_dir_entry_list:
		model_shapenet_dir = os.path.join(shapenet_synset_dir, shapenet_synset_dir_entry)
		if (not os.path.isdir(model_shapenet_dir)):
			continue
		
		model_shapenet_id = shapenet_synset_dir_entry
		model_shapenet_path = os.path.join(model_shapenet_dir, "models", "model_normalized.obj")
		
		if (not os.path.isfile(model_shapenet_path)):
			continue
		 
		model_output_dir = os.path.join(synset_output_dir, model_shapenet_id)

		render_model_command = "%s --background --python %s -- %s %s %s %s %d" % (
			global_vars.BLENDER_EXECUTABLE_PATH, os.path.join(this_file_directory, 'render_model.py'), 
			model_shapenet_id, model_shapenet_path, model_output_dir, synset_meta_file_path, synset_meta_header_length 
		)

		print("\nRendering model %d out of %d" %(model_index, len(shapenet_synset_dir_entry_list)))
		print("Command: %s\n" %(render_model_command, ))

		os.system(render_model_command)

		model_index = model_index + 1
		# if (model_index > 20):
		# 	break


if __name__ == "__main__":
	main()