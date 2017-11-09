##
# \file simulate_stacks_from_reconstruction.py
# \brief      Simulate stacks from obtained reconstruction
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       November 2017
#

# Import libraries
import os
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.stack as st
import niftymic.base.data_reader as dr
import niftymic.base.data_writer as dw
import niftymic.reconstruction.linear_operators as lin_op
from niftymic.utilities.input_arparser import InputArgparser


def main():

    input_parser = InputArgparser(
        description="Simulate stacks from obtained reconstruction",
    )
    input_parser.add_dir_input(required=True)
    input_parser.add_reconstruction(required=True)
    input_parser.add_dir_output(required=True)
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_prefix_output(default="Simulated_")
    input_parser.add_option(
        option_string="--copy-data",
        type=int,
        help="Turn on/off copying of original data (including masks) to "
        "output folder.",
        default=0)
    input_parser.add_verbose(default=0)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    data_reader = dr.ImageSlicesDirectoryReader(
        path_to_directory=args.dir_input,
        suffix_mask=args.suffix_mask)
    data_reader.read_data()
    stacks = data_reader.get_stacks()

    reconstruction = st.Stack.from_filename(
        args.reconstruction, extract_slices=False)

    linear_operators = lin_op.LinearOperators()

    for i, stack in enumerate(stacks):

        # initialize data array
        nda = sitk.GetArrayFromImage(stack.sitk) * 0
        simulated_slices = [
            linear_operators.A(reconstruction, s) for s in stack.get_slices()
        ]

        for j, simulated_slice in enumerate(simulated_slices):
            nda[j, :, :] = sitk.GetArrayFromImage(simulated_slice.sitk)

        simulated_stack_sitk = sitk.GetImageFromArray(nda)
        simulated_stack_sitk.CopyInformation(stack.sitk)

        simulated_stack = st.Stack.from_sitk_image(
            image_sitk=simulated_stack_sitk,
            filename=args.prefix_output + stack.get_filename(),
            extract_slices=False)

        if args.verbose:
            sitkh.show_stacks([stack, simulated_stack], segmentation=stack)

        simulated_stack.write(
            args.dir_output, write_mask=False, write_slices=False)
        stack.write(
            args.dir_output, write_mask=True, write_slices=False)

    return 0


if __name__ == '__main__':
    main()
