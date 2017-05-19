#!/usr/bin/python

## \file reconstructStaticVolume.py
#  \brief  Script to reconstruct an isotropic, high-resolution volume from 
#  multiple stacks of low-resolution 2D slices. Motion-correction is NOT
#  performed in this file.
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2017

## Import libraries 
import SimpleITK as sitk
import argparse

import numpy as np
import sys
import os

## Import modules
import utilities.SimpleITKHelper as sitkh
import utilities.PythonHelper as ph
import preprocessing.DataPreprocessing as dp
import base.Stack as st
import registration.SegmentationPropagation as segprop
import reconstruction.solver.TikhonovSolver as tk


##
# Gets the parsed input line.
# \date       2017-05-18 20:09:23+0100
#
# \param      dir_output          The dir output
# \param      prefix_output       The prefix output
# \param      suffix_mask         The suffix mask
# \param      target_stack_index  The target stack index
# \param      regularization      The regularization
# \param      minimizer           The minimizer
# \param      alpha               The alpha
# \param      iter_max            The iterator maximum
# \param      verbose             The verbose
#
# \return     The parsed input line.
#
def get_parsed_input_line(
    dir_output,
    prefix_output,
    suffix_mask,
    target_stack_index, 
    regularization, 
    alpha,
    iter_max,
    verbose,
    comparison_script,
    ):

    parser = argparse.ArgumentParser(description=
        "Volumetric reconstruction framework to reconstruct an isotropic, high-resolution 3D volume from multiple stacks of 2D slices. The final resolution is given by the in-plane resolution of the selected target stack. "
        "In case a mask is provided for the selected target stack, it will be propagated to the remaining stacks. This will specify the field-of-view and accelerate the computational time.",
        version="0.1",
        usage="python reconstructStaticVolume.py --dir_input=path-to-input-directory --dir_output=path-to-output-directory --iter_max=10 --alpha=0.02 --suffix_mask=_mask",
        epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)"
        )

    parser.add_argument('--dir_input', type=str, help="Input directory with NIfTI files (.nii or .nii.gz).", required=True)
    parser.add_argument('--dir_output', type=str, help="Output directory. [default: %s]" %(dir_output), default=dir_output)
    parser.add_argument('--suffix_mask', type=str, help="Suffix used to specify the mask. E.g. image.nii.gz with associated mask image_mask.nii.gz. [default: %s]" %(suffix_mask), default=suffix_mask)
    parser.add_argument('--prefix_output', type=str, help="Prefix for SRR output. [default: %s]" %(prefix_output), default=prefix_output)
    parser.add_argument('--target_stack_index', type=int, help="Index of stack (image) in input directory (alphabetical order) which defines physical space for SRR. First index is 0. [default: %s]" %(target_stack_index), default=target_stack_index)
    parser.add_argument('--regularization', type=str, help="Type of regularization for SR algorithm. Either 'TK0' or 'TK1' for zeroth or first order Tikhonov regularization. [default: %s]" %(regularization), default=regularization)
    parser.add_argument('--alpha', type=float, help="Regularization parameter alpha to solve reconstruction problem sum_k ||y_k - A_k x|| + alpha R(x). [default: %g]" %(alpha), default=alpha)
    parser.add_argument('--iter_max', type=int, help="Number of maximum iterations for numerical solver", default=iter_max)
    parser.add_argument('--verbose', type=bool, help="Turn on verbose. [default: %s]" %(verbose), default=verbose)
    parser.add_argument('--comparison_script', type=bool, help="Generate a comparison script to provide comparison of SRR with (linearly resampled) original data. [default: %s]" %(comparison_script), default=comparison_script)

    args = parser.parse_args()

    if args.verbose:
        ph.print_title("Given Input")
        print("Set Parameters:")
        for arg in sorted(vars(args)):
            ph.print_debug_info("%s: " %(arg), newline=False)
            print(getattr(args, arg))

    return args


"""
Main Function
"""
if __name__ == '__main__':

    time_start = ph.start_timing()

    ##-------------------------------------------------------------------------
    ## Read input
    args = get_parsed_input_line(
        dir_output="/tmp/",
        prefix_output="SRR",
        suffix_mask="_mask",
        target_stack_index=0,
        regularization="TK1",
        alpha=0.02,
        iter_max=10,
        verbose=1,
        comparison_script=1,
        )

    ##-------------------------------------------------------------------------
    ## Data Preprocessing from data on HDD
    ph.print_title("Data Preprocessing")
    segmentation_propagator = segprop.SegmentationPropagation(
        dilation_radius=3,
        dilation_kernel="Ball",
        )

    data_preprocessing = dp.DataPreprocessing.from_directory(
        dir_input=args.dir_input, 
        suffix_mask=args.suffix_mask,
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
    stacks = data_preprocessing.get_preprocessed_stacks()

    # sitkh.show_stacks(stacks)

    ##-------------------------------------------------------------------------
    ## SRR
    ph.print_title("Super-Resolution Reconstruction")
    
    ## Initial value specifying the physical space for the HR reconstruction.
    ## In-plane spacing of chosen template stack defines isotropic voxel size.
    HR_volume_init = stacks[0].get_isotropically_resampled_stack()
    HR_volume_init.set_filename("HR_volume_0")

    ## SRR step
    HR_volume = st.Stack.from_stack(HR_volume_init, filename="HR_volume")
    SRR = tk.TikhonovSolver(
        stacks=stacks,
        HR_volume=HR_volume,
        reg_type=args.regularization,
        minimizer="lsmr",
        iter_max=args.iter_max,
        alpha=args.alpha,
        )
    SRR.run_reconstruction()
    SRR.print_statistics()
    
    time_SRR = SRR.get_computational_time()
    elapsed_time = ph.stop_timing(time_start)

    ## Update filename
    filename = SRR.get_setting_specific_filename(prefix=args.prefix_output)
    HR_volume.set_filename(filename) 
    # HR_volume.show()
    
    ##-------------------------------------------------------------------------
    ## Write SRR to output
    HR_volume.write(directory=args.dir_output)

    ## Show SRR together with linearly resampled input data.
    ## Additionally, a script is generated to open files
    stacks_visualization = []
    stacks_visualization.append(HR_volume)
    for i in range(0, len(stacks)):
        stacks_visualization.append(stacks[i])stacks
    
    sitkh.show_stacks(stacks_visualization, 
        show_comparison_file=comparison_script,
        dir_output=args.dir_output,
        )

    ##-------------------------------------------------------------------------
    ## Summary
    ph.print_title("Summary")
    print("Computational Time for Data Preprocessing: %s" %(time_data_preprocessing))
    print("Computational Time for Super-Resolution Algorithm: %s" %(time_SRR))
    print("Computational Time for Entire Reconstruction Pipeline: %s" %(elapsed_time))
