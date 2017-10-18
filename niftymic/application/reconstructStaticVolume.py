#!/usr/bin/python

##
# \file reconstructStaticVolume.py
# \brief      Script to reconstruct an isotropic, high-resolution volume from
#             multiple stacks of low-resolution 2D slices without
#             motion-correction.
#
# Example usage:
#       - `python reconstructStaticVolume.py --help`
#       - `python reconstructStaticVolume.py --dir_input=path-to-data`
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       May 2017
#

# Import libraries
import SimpleITK as sitk
import argparse
import numpy as np
import sys
import os

import pysitk.simple_itk_helper as sitkh
import pysitk.python_helper as ph

# Import modules
import niftymic.base.Stack as st
import niftymic.base.DataReader as dr
import niftymic.preprocessing.DataPreprocessing as dp
import niftymic.registration.SegmentationPropagation as segprop
import niftymic.reconstruction.solver.TikhonovSolver as tk
from niftymic.utilities.InputArparser import InputArgparser
import niftymic.utilities.Exceptions as Exceptions


##
# Main Function
#
if __name__ == '__main__':

    time_start = ph.start_timing()

    # Set print options for numpy
    np.set_printoptions(precision=3)

    # Read input
    input_parser = InputArgparser(
        description="Volumetric MRI reconstruction framework to reconstruct "
        "an isotropic, high-resolution 3D volume from multiple stacks of 2D "
        "slices WITHOUT motion correction. The resolution of the computed "
        "Super-Resolution Reconstruction (SRR) is given by the in-plane "
        "spacing of the selected target stack. A region of interest can be "
        "specified by providing a mask for the selected target stack. Only "
        "this region will then be reconstructed by the SRR algorithm which "
        "can substantially reduce the computational time.",
        prog="python " + os.path.basename(__file__),
    )
    input_parser.add_dir_input()
    input_parser.add_filenames()
    input_parser.add_dir_output(default="results/")
    input_parser.add_prefix_output(default="_SRR")
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_target_stack_index(default=0)
    input_parser.add_reg_type(default="TK1")
    input_parser.add_alpha(default=0.02)
    input_parser.add_iter_max(default=10)
    input_parser.add_provide_comparison(default=0)
    input_parser.add_log_script_execution(default=1)
    input_parser.add_verbose(default=0)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # Write script execution call
    if args.log_script_execution:
        input_parser.write_performed_script_execution(
            os.path.abspath(__file__))

    # Read Data:
    ph.print_title("Read Data")

    # Neither '--dir-input' nor '--filenames' was specified
    if args.filenames is not None and args.dir_input is not None:
        raise Exceptions.IOError(
            "Provide input by either '--dir-input' or '--filenames' "
            "but not both together")

    # '--dir-input' specified
    elif args.dir_input is not None:
        data_reader = dr.DirectoryReader(
            args.dir_input, suffix_mask=args.suffix_mask)

    # '--filenames' specified
    elif args.filenames is not None:
        data_reader = dr.MultipleImagesReader(
            args.filenames, suffix_mask=args.suffix_mask)

    else:
        raise Exceptions.IOError(
            "Provide input by either '--dir-input' or '--filenames'")

    data_reader.read_data()
    stacks = data_reader.get_stacks()

    # Data Preprocessing
    ph.print_title("Data Preprocessing")
    segmentation_propagator = segprop.SegmentationPropagation(
        dilation_radius=3,
        dilation_kernel="Ball",
    )

    data_preprocessing = dp.DataPreprocessing(
        stacks=stacks,
        segmentation_propagator=segmentation_propagator,
        use_cropping_to_mask=True,
        target_stack_index=args.target_stack_index,
        boundary_i=0,
        boundary_j=0,
        boundary_k=0,
        unit="mm",
    )
    data_preprocessing.run_preprocessing()
    time_data_preprocessing = data_preprocessing.get_computational_time()

    # Get preprocessed stacks
    stacks = data_preprocessing.get_preprocessed_stacks()

    if args.verbose:
        sitkh.show_stacks(stacks, segmentation=stacks[args.target_stack_index])

    # Super-Resolution Reconstruction (SRR)
    ph.print_title("Super-Resolution Reconstruction")

    ##
    # Initial, isotropic volume to define the physical space for the HR SRR
    # reconstruction. In-plane spacing of chosen template stack defines
    # the isotropic voxel size.
    HR_volume_init = stacks[args.target_stack_index].\
        get_isotropically_resampled_stack()
    HR_volume_init.set_filename("HR_volume_0")

    # SRR step
    HR_volume = st.Stack.from_stack(HR_volume_init, filename="HR_volume")
    SRR = tk.TikhonovSolver(
        stacks=stacks,
        reconstruction=HR_volume,
        reg_type=args.reg_type,
        iter_max=args.iter_max,
        alpha=args.alpha,
    )
    SRR.run_reconstruction()
    SRR.print_statistics()

    time_SRR = SRR.get_computational_time()
    elapsed_time = ph.stop_timing(time_start)

    # Update filename
    filename = SRR.get_setting_specific_filename(prefix=args.prefix_output)
    HR_volume.set_filename(filename)

    if args.verbose:
        HR_volume.show()

    # Write SRR to output
    HR_volume.write(directory=args.dir_output)

    # Show SRR together with linearly resampled input data.
    # Additionally, a script is generated to open files
    if args.provide_comparison or args.verbose:
        stacks_visualization = []
        stacks_visualization.append(HR_volume)
        for i in range(0, len(stacks)):
            stacks_visualization.append(stacks[i])

        sitkh.show_stacks(stacks_visualization,
                          show_comparison_file=args.provide_comparison,
                          dir_output=os.path.join(
                              args.dir_output, "comparison"),
                          )

    # Summary
    ph.print_title("Summary")
    print("Computational Time for Data Preprocessing: %s" %
          (time_data_preprocessing))
    print("Computational Time for Super-Resolution Algorithm: %s" % (time_SRR))
    print("Computational Time for Entire Reconstruction Pipeline: %s" %
          (elapsed_time))
