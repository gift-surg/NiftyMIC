#!/usr/bin/python

##
# \file correct_bias_field.py
# \brief      Script to correct for bias field. Based on N4ITK
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       October 2017
#

# Import libraries
import SimpleITK as sitk
import argparse
import numpy as np
import sys
import os

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

# Import modules
import niftymic.base.data_reader as dr
import niftymic.base.stack as st
import niftymic.preprocessing.n4_bias_field_correction as n4itk
from niftymic.utilities.input_arparser import InputArgparser


def main():

    time_start = ph.start_timing()

    np.set_printoptions(precision=3)

    input_parser = InputArgparser(
        description="Perform Bias Field correction on images based on N4ITK.",
    )
    input_parser.add_filenames(required=True)
    input_parser.add_dir_output(required=True)
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_prefix_output(default="N4ITK_")
    input_parser.add_option(
        option_string="--convergence-threshold",
        type=float,
        help="Specify the convergence threshold.",
        default=1e-6,
    )
    input_parser.add_option(
        option_string="--spline-order",
        type=int,
        help="Specify the spline order defining the bias field estimate.",
        default=3,
    )
    input_parser.add_option(
        option_string="--wiener-filter-noise",
        type=float,
        help="Specify the noise estimate defining the Wiener filter.",
        default=0.11,
    )
    input_parser.add_option(
        option_string="--bias-field-fwhm",
        type=float,
        help="Specify the full width at half maximum parameter characterizing "
        "the width of the Gaussian deconvolution.",
        default=0.15,
    )
    input_parser.add_log_script_execution(default=1)
    input_parser.add_verbose(default=0)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # Write script execution call
    if args.log_script_execution:
        input_parser.write_performed_script_execution(
            os.path.abspath(__file__))

    # Read data
    data_reader = dr.MultipleImagesReader(
        args.filenames, suffix_mask=args.suffix_mask)
    data_reader.read_data()
    stacks = data_reader.get_stacks()

    # Perform Bias Field Correction
    ph.print_title("Perform Bias Field Correction")
    bias_field_corrector = n4itk.N4BiasFieldCorrection(
        convergence_threshold=args.convergence_threshold,
        spline_order=args.spline_order,
        wiener_filter_noise=args.wiener_filter_noise,
        bias_field_fwhm=args.bias_field_fwhm,
        prefix_corrected=args.prefix_output,
    )
    stacks_corrected = [None] * len(stacks)
    for i, stack in enumerate(stacks):
        ph.print_info("Image %d/%d: N4ITK Bias Field Correction ... "
                      % (i+1, len(stacks)), newline=False)
        bias_field_corrector.set_stack(stack)
        bias_field_corrector.run_bias_field_correction()
        stacks_corrected[i] = \
            bias_field_corrector.get_bias_field_corrected_stack()
        print("done")
        ph.print_info("Image %d/%d: Computational time = %s"
                      % (i+1,
                         len(stacks),
                         bias_field_corrector.get_computational_time()))

        # Write Data
        stacks_corrected[i].write(
            args.dir_output, write_mask=True, suffix_mask=args.suffix_mask)

        if args.verbose:
            sitkh.show_stacks([stacks[i], stacks_corrected[i]],
                              segmentation=stacks[i])

    elapsed_time = ph.stop_timing(time_start)

    ph.print_title("Summary")
    print("Computational Time for Bias Field Correction(s): %s" %
          (elapsed_time))

    return 0

if __name__ == '__main__':
    main()
