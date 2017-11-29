##
# \file visualize_slice_select_directions.py
# \brief      Script to visualize slice select directions of acquired image stacks.
#
# This function takes the result of simulate_stacks_from_reconstruction.py as
# input.
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       November 2017
#

# Import libraries
import SimpleITK as sitk
import numpy as np
import os

import pysitk.python_helper as ph
from nsol.similarity_measures import SimilarityMeasures as \
    SimilarityMeasures

import niftymic.base.data_reader as dr
import niftymic.prototyping.stacks_visualizer as sv
from niftymic.utilities.input_arparser import InputArgparser


def main():

    # Read input
    input_parser = InputArgparser(
        description="Script to evaluate the similarity of simulated stack "
        "from obtained reconstruction against the original stack. "
        "This function takes the result of "
        "simulate_stacks_from_reconstruction.py as input.",
    )
    input_parser.add_filenames(required=True)
    input_parser.add_dir_output(required=False)
    input_parser.add_labels(default=None)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    stacks_visualizer = sv.StacksVisualizer.from_filenames(
        args.filenames, labels=args.labels)
    fig = stacks_visualizer.show_slice_select_directions()

    if args.dir_output is not None:
        ph.save_fig(fig, args.dir_output, "slice_select_directions.pdf")

    return 0


if __name__ == '__main__':
    main()