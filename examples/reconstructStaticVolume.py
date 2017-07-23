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

# Import modules
sys.path.insert(1, os.path.abspath(
    os.path.join(os.environ['VOLUMETRIC_RECONSTRUCTION_DIR'], 'src', 'py')))
import base.Stack as st
import base.DataReader as dr
import pythonhelper.SimpleITKHelper as sitkh
import pythonhelper.PythonHelper as ph
import preprocessing.DataPreprocessing as dp
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
    filenames,
    prefix_output,
    suffix_mask,
    target_stack_index,
    regularization,
    alpha,
    iter_max,
    verbose,
    provide_comparison,
    log_script_execution,
):

    parser = argparse.ArgumentParser(
        description="Volumetric MRI reconstruction framework to reconstruct "
        "an isotropic, high-resolution 3D volume from multiple stacks of 2D "
        "slices WITHOUT motion correction. The resolution of the computed "
        "Super-Resolution Reconstruction (SRR) is given by the in-plane "
        "spacing of the selected target stack. A region of interest can be "
        "specified by providing a mask for the selected target stack. Only "
        "this region will then be reconstructed by the SRR algorithm which "
        "can substantially reduce the computational time.",
        prog="python reconstructStaticVolume.py",
        epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
    )

    parser.add_argument('--dir-input',
                        type=str,
                        help="Input directory with NIfTI files "
                        "(.nii or .nii.gz).",
                        default="")
    parser.add_argument('--filenames',
                        nargs="+",
                        help="Filenames. [default: %s]" % (filenames),
                        default=filenames)
    parser.add_argument('--dir-output',
                        type=str,
                        help="Output directory. [default: %s]" % (dir_output),
                        default=dir_output)
    parser.add_argument('--suffix-mask',
                        type=str,
                        help="Suffix used to associate a mask with an image. "
                        "E.g. suffix_mask='_mask' means an existing "
                        "image_i_mask.nii.gz represents the mask to "
                        "image_i.nii.gz for all images image_i in the input "
                        "directory. [default: %s]" % (suffix_mask),
                        default=suffix_mask)
    parser.add_argument('--prefix-output',
                        type=str,
                        help="Prefix for SRR output file name. [default: %s]"
                        % (prefix_output),
                        default=prefix_output)
    parser.add_argument('--target-stack-index',
                        type=int,
                        help="Index of stack (image) in input directory "
                        "(alphabetical order) which defines physical space "
                        "for SRR. First index is 0. [default: %s]"
                        % (target_stack_index),
                        default=target_stack_index)
    parser.add_argument('--alpha',
                        type=float,
                        help="Regularization parameter alpha to solve the SR "
                        "reconstruction problem: SRR = argmin_x "
                        "[0.5 * sum_k ||y_k - A_k x||^2 + alpha * R(x)]. "
                        "[default: %g]" % (alpha),
                        default=alpha)
    parser.add_argument('--regularization',
                        type=str,
                        help="Type of regularization for SR algorithm. Either "
                        "'TK0' or 'TK1' for zeroth or first order Tikhonov "
                        "regularization, respectively. I.e. R(x) = ||x||^2 "
                        "for 'TK0' or R(x) = ||Dx||^2 for 'TK1'. [default: %s]"
                        % (regularization),
                        default=regularization)
    parser.add_argument('--iter-max',
                        type=int,
                        help="Number of maximum iterations for the numerical "
                        "solver. [default: %s]" % (iter_max),
                        default=iter_max)
    parser.add_argument('--log-script-execution',
                        type=int,
                        help="Turn on/off log for execution of current "
                        "script. [default: %s]" % (log_script_execution),
                        default=log_script_execution)
    parser.add_argument('--verbose',
                        type=int,
                        help="Turn on/off verbose output. [default: %s]"
                        % (verbose),
                        default=verbose)
    parser.add_argument('--provide-comparison',
                        type=int,
                        help="Turn on/off functionality to create files "
                        "allowing for a visual comparison between original "
                        "data and the obtained SRR. A folder 'comparison' "
                        "will be created in the output directory containing "
                        "the obtained SRR along with the linearly resampled "
                        "original data. An additional script "
                        "'show_comparison.py' will be provided whose "
                        "execution will open all images in ITK-Snap "
                        "(http://www.itksnap.org/). [default: %s]"
                        % (provide_comparison),
                        default=provide_comparison)

    args = parser.parse_args()

    ph.print_title("Given Input")
    print("Chosen Parameters:")
    for arg in sorted(vars(args)):
        ph.print_debug_info("%s: " % (arg), newline=False)
        print(getattr(args, arg))

    return args


##
# Main Function
#
if __name__ == '__main__':

    time_start = ph.start_timing()

    # Set print options for numpy
    np.set_printoptions(precision=3)

    # Read input
    args = get_parsed_input_line(
        dir_output="results/",
        filenames="",
        prefix_output="SRR_",
        suffix_mask="_mask",
        target_stack_index=0,
        regularization="TK1",
        alpha=0.02,
        iter_max=10,
        provide_comparison=0,
        log_script_execution=1,
        verbose=0,
    )

    # Write script execution call
    if args.log_script_execution:
        performed_script_execution = ph.get_performed_script_execution(
            os.path.basename(__file__), args)
        ph.write_performed_script_execution_to_executable_file(
            performed_script_execution,
            os.path.join(args.dir_output, "log_script_execution.sh"))

    # Read Data:
    ph.print_title("Read Data")

    # Neither '--dir-input' nor '--filenames' was specified
    if args.filenames != "" and args.dir_input != "":
        raise Exceptions.IOError(
            "Provide input by either '--dir-input' or '--filenames' "
            "but not both together")

    # '--dir-input' specified
    elif args.dir_input != "":
        data_reader = dr.DirectoryReader(
            args.dir_input, suffix_mask=args.suffix_mask)

    # '--filenames' specified
    else:
        data_reader = dr.MultipleImagesReader(
            args.filenames[0], suffix_mask=args.suffix_mask)

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

    # Get preprocessed stacks with index 0 holding the selected target stack
    stacks = data_preprocessing.get_preprocessed_stacks()

    if args.verbose:
        sitkh.show_stacks(stacks, segmentation=stacks[0])

    # Super-Resolution Reconstruction (SRR)
    ph.print_title("Super-Resolution Reconstruction")

    ##
    # Initial, isotropic volume to define the physical space for the HR SRR
    # reconstruction. In-plane spacing of chosen template stack defines
    # the isotropic voxel size.
    HR_volume_init = stacks[0].get_isotropically_resampled_stack()
    HR_volume_init.set_filename("HR_volume_0")

    # SRR step
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
