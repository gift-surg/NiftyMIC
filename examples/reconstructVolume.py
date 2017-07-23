#!/usr/bin/python

##
# \file reconstructVolume.py
# \brief      Script to reconstruct an isotropic, high-resolution volume from
#             multiple stacks of low-resolution 2D slices including
#             motion-correction.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       March 2016
#

# Import libraries
import SimpleITK as sitk
import argparse
import inspect
from inspect import getframeinfo, stack

import traceback
import numpy as np
import sys
import os

# Add directories to import modules
sys.path.insert(1, os.path.abspath(os.path.join(
    os.environ['VOLUMETRIC_RECONSTRUCTION_DIR'], 'src', 'py')))

# Import modules
import base.DataReader as dr
import base.Stack as st
import preprocessing.DataPreprocessing as dp
import preprocessing.N4BiasFieldCorrection as n4bfc
import registration.RegistrationSimpleITK as regsitk
import registration.RegistrationITK as regitk
import registration.NiftyReg as regniftyreg
import registration.SegmentationPropagation as segprop
import reconstruction.ScatteredDataApproximation as sda
import reconstruction.solver.TikhonovSolver as tk
import reconstruction.solver.ADMMSolver as admm
import pythonhelper.PythonHelper as ph
import pythonhelper.SimpleITKHelper as sitkh
import utilities.Exceptions as Exceptions


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
    two_step_cycles,
    sigma,
    regularization,
    loss,
    alpha,
    alpha_final,
    iter_max,
    iter_max_final,
    minimizer,
    dilation_radius,
    extra_frame_target,
    bias_field_correction,
    intensity_correction,
    provide_comparison,
    isotropic_resolution,
    log_script_execution,
    verbose,
):

    parser = argparse.ArgumentParser(
        description="Volumetric MRI reconstruction framework to reconstruct "
        "an isotropic, high-resolution 3D volume from multiple stacks of 2D "
        "slices WITH motion correction. The resolution of the computed "
        "Super-Resolution Reconstruction (SRR) is given by the in-plane "
        "spacing of the selected target stack. A region of interest can be "
        "specified by providing a mask for the selected target stack. Only "
        "this region will then be reconstructed by the SRR algorithm which "
        "can substantially reduce the computational time.",
        prog="python reconstructVolume.py",
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
    parser.add_argument('--sigma',
                        type=float,
                        help="Standard deviation for Scattered Data "
                        "Approximation approach to reconstruct first estimate "
                        "of HR volume from all 3D input stacks. [default: %g]"
                        % (sigma),
                        default=sigma)
    parser.add_argument('--alpha',
                        type=float,
                        help="Regularization parameter alpha to solve the SR "
                        "reconstruction problem: SRR = argmin_x "
                        "[0.5 * sum_k ||y_k - A_k x||^2 + alpha * R(x)]. "
                        "[default: %g]" % (alpha),
                        default=alpha)
    parser.add_argument('--alpha-final',
                        type=float,
                        help="Regularization parameter like 'alpha' but used "
                        "for the final SRR step. [default: %s]"
                        % (alpha_final),
                        default=alpha_final)
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
    parser.add_argument('--iter-max-final',
                        type=int,
                        help="Number of maximum iterations for the numerical "
                        "solver like 'iter-max' but used for the final SRR "
                        "step [default: %s]" % (iter_max_final),
                        default=iter_max_final)
    parser.add_argument('--minimizer',
                        type=str,
                        help="Choice of minimizer used for the inverse "
                        "problem associated to the SRR. Possible choices are "
                        "'lsmr' or 'L-BFGS-B'. [default: %s]" % (minimizer),
                        default=minimizer)
    parser.add_argument('--two-step-cycles',
                        type=int,
                        help="Number of two-step-cycles, i.e. number of "
                        "Slice-to-Volume Registration and Super-Resolution "
                        "Reconstruction cycles. [default: %s]"
                        % (two_step_cycles),
                        default=two_step_cycles)
    parser.add_argument('--loss',
                        type=str,
                        help="Loss function rho used for data term, i.e. "
                        "rho((y_k - A_k x)^2). Possible choices are 'linear', "
                        "'soft_l1' or 'huber'. [default: %s]" % (loss),
                        default=loss)
    parser.add_argument('--dilation-radius',
                        type=int,
                        help="Dilation radius in number of voxels used for "
                        "segmentation propagation. [default: %s]"
                        % (dilation_radius),
                        default=dilation_radius)
    parser.add_argument('--extra-frame-target',
                        type=float,
                        help="Extra frame in mm added to the increase the "
                        "target space for the SRR. [default: %s]"
                        % (extra_frame_target),
                        default=extra_frame_target)
    parser.add_argument('--bias-field-correction',
                        type=int,
                        help="Turn on/off bias field correction during data "
                        "preprocessing. [default: %s]"
                        % (bias_field_correction),
                        default=bias_field_correction)
    parser.add_argument('--intensity-correction',
                        type=int,
                        help="Turn on/off linear intensity correction during "
                        "data preprocessing. [default: %s]"
                        % (intensity_correction),
                        default=intensity_correction)
    parser.add_argument('--isotropic-resolution',
                        type=float,
                        help="Specify isotropic resolution for obtained HR "
                        "volume. [default: %s]" % (isotropic_resolution),
                        default=isotropic_resolution)
    parser.add_argument('--log-script-execution',
                        type=int,
                        help="Turn on/off log for execution of current "
                        "script. [default: %s]" % (log_script_execution),
                        default=log_script_execution)
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
    parser.add_argument('--verbose',
                        type=int,
                        help="Turn on/off verbose output. [default: %s]"
                        % (verbose),
                        default=verbose)
    args = parser.parse_args()

    ph.print_title("Given Input")
    print("Chosen Parameters:")
    for arg in sorted(vars(args)):
        ph.print_debug_info("%s: " % (arg), newline=False)
        print(getattr(args, arg))

    return args

"""
Main Function
"""
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
        two_step_cycles=3,
        sigma=0.7,
        regularization="TK1",
        loss="linear",
        alpha=0.1,
        alpha_final=0.03,
        isotropic_resolution=None,
        iter_max=5,
        iter_max_final=10,
        minimizer="lsmr",
        dilation_radius=3,
        extra_frame_target=10,
        bias_field_correction=0,
        intensity_correction=0,
        provide_comparison=1,
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

    # Data Preprocessing from data on HDD
    ph.print_title("Data Preprocessing")

    segmentation_propagator = segprop.SegmentationPropagation(
        # registration_method=regniftyreg.NiftyReg(use_verbose=args.verbose),
        # registration_method=regsitk.RegistrationSimpleITK(use_verbose=args.verbose),
        # registration_method=regitk.RegistrationITK(use_verbose=args.verbose),
        dilation_radius=args.dilation_radius,
        dilation_kernel="Ball",
    )

    data_preprocessing = dp.DataPreprocessing(
        stacks=stacks,
        segmentation_propagator=segmentation_propagator,
        use_cropping_to_mask=True,
        use_N4BiasFieldCorrector=args.bias_field_correction,
        use_intensity_correction=args.intensity_correction,
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
        for i in range(0, len(stacks)):
            stacks[i].write(
                directory=os.path.join(
                    args.dir_output, "01_preprocessed_data"),
                write_mask=True)

    # # ----------------- Begin HACK for Imperial College data ----------------
    # # Split stack acquired as overlapping slices into two
    # stacks_foo = []
    # for i in range(0, len(stacks)):
    #     for j in range(0, 2):
    #         stack_sitk = stacks[i].sitk[:, :, j::2]
    #         stack_sitk_mask = stacks[i].sitk_mask[:, :, j::2]
    #         stacks_foo.append(
    #             st.Stack.from_sitk_image(stack_sitk,
    #                                      image_sitk_mask=stack_sitk_mask,
    #                                      filename=stacks[i].get_filename() +
    #                                      "_" + str(j+1)))
    # stacks = stacks_foo
    # # ------------------ End HACK for Imperial College data -----------------

    if args.verbose:
        sitkh.show_stacks(stacks, segmentation=stacks[0])

    if args.two_step_cycles > 0:

        time_registration = ph.start_timing()

        # Global rigid registration to target stack
        ph.print_title("Global Rigid Registration")

        # registration = regsitk.RegistrationSimpleITK(
        #     initializer_type="GEOMETRY", metric="MattesMutualInformation",
        registration = regniftyreg.NiftyReg(
            fixed=stacks[0],
            registration_type="Rigid",
            use_fixed_mask=True,
            use_moving_mask=True,
            use_verbose=False,
        )

        for i in range(1, len(stacks)):
            registration.set_moving(stacks[i])
            registration.run_registration()
            transform_sitk = registration.get_registration_transform_sitk()
            transform_sitk = eval(
                "sitk." + transform_sitk.GetName() +
                "(transform_sitk.GetInverse())")
            stacks[i].update_motion_correction(transform_sitk)

        time_registration = ph.stop_timing(time_registration)
        # sitkh.show_stacks(stacks, segmentation=stacks[0])

        if args.verbose:
            for i in range(0, len(stacks)):
                stacks[i].write(
                    directory=os.path.join(args.dir_output,
                                           "02_rigidly_aligned_data"),
                    write_mask=True)

    else:
        time_registration = 0

    time_reconstruction = ph.start_timing()

    # Isotropic resampling to define HR target space
    ph.print_title("Isotropic Resampling")
    HR_volume = stacks[0].get_isotropically_resampled_stack(
        spacing_new_scalar=args.isotropic_resolution,
        extra_frame=args.extra_frame_target)
    HR_volume.set_filename(stacks[0].get_filename() + "_upsampled")

    # Scattered Data Approximation to get first estimate of HR volume
    ph.print_title("Scattered Data Approximation")
    SDA = sda.ScatteredDataApproximation(stacks, HR_volume, sigma=args.sigma)
    SDA.run_reconstruction()
    SDA.generate_mask_from_stack_mask_unions(
        mask_dilation_radius=2, mask_dilation_kernel="Ball")
    HR_volume = SDA.get_reconstruction()
    HR_volume.set_filename("HRvolume_SDA")

    time_reconstruction = ph.stop_timing(time_reconstruction)

    # List to store SRR iterations
    HR_volume_iterations = []

    # Add initial volume and rigidly aligned, original data for
    # visualization
    HR_volume_iterations.append(
        st.Stack.from_stack(HR_volume, "HRvolume_iter0"))
    for i in range(0, len(stacks)):
        HR_volume_iterations.append(stacks[i])

    if args.verbose:
        sitkh.show_stacks(HR_volume_iterations)

    if args.regularization in ["TK0", "TK1"]:
        SRR = tk.TikhonovSolver(
            stacks=stacks,
            HR_volume=HR_volume,
            alpha=args.alpha,
            iter_max=args.iter_max,
            reg_type=args.regularization,
            minimizer=args.minimizer,
            loss=args.loss,
        )
    elif args.regularization == "TV":
        SRR = admm.ADMMSolver(
            stacks=stacks,
            HR_volume=HR_volume,
            alpha=args.alpha,
            reg_type=args.regularization,
            minimizer=args.minimizer,
            iter_max=args.iter_max,
            rho=rho,
            ADMM_iterations=ADMM_iterations,
        )

    if args.two_step_cycles > 0:

        alpha_delta = (args.alpha_final - args.alpha) / \
            float(args.two_step_cycles)

        # Two-step Slice-to-Volume Registration Reconstruction
        ph.print_title("Two-step Slice-to-Volume Registration Reconstruction")

        # Two-step registration reconstruction
        registration = regsitk.RegistrationSimpleITK(
            moving=HR_volume,
            use_fixed_mask=True,
            use_moving_mask=False,
            use_verbose=True,
            interpolator="Linear",
            metric="Correlation",
            # metric="MattesMutualInformation",  # Might cause error messages
            # like "Too many samples map outside moving image buffer."
            # use_multiresolution_framework=True,
            initializer_type=None,
            # optimizer="RegularStepGradientDescent",
            # optimizer_params="{'learningRate': 1, 'minStep': 1e-6,\
            # 'numberOfIterations': 600, 'gradientMagnitudeTolerance': 1e-6}",
            optimizer="ConjugateGradientLineSearch",
            optimizer_params="{'learningRate': 1, 'numberOfIterations': 100}",
        )

        for i_cycle in range(0, args.two_step_cycles):
            time_elapsed_tmp = ph.start_timing()
            for i_stack in range(0, len(stacks)):
                stack = stacks[i_stack]

                # Slice-to-volume registration
                for i_slice in range(0, stack.get_number_of_slices()):
                    ph.print_subtitle(
                        "Cycle %d/%d -- Stack %d/%d -- Slice %2d/%d: "
                        "Slice-to-Volume Registration" % (
                            i_cycle+1, args.two_step_cycles, i_stack+1,
                            len(stacks), i_slice+1,
                            stack.get_number_of_slices()))
                    slice = stack.get_slice(i_slice)
                    registration.set_fixed(slice)
                    registration.run_registration()
                    transform_sitk = \
                        registration.get_registration_transform_sitk()
                    slice.update_motion_correction(transform_sitk)
            time_elapsed_tmp = ph.stop_timing(time_elapsed_tmp)
            time_registration = ph.add_times(
                time_registration, time_elapsed_tmp)
            print("\nElapsed time for all Slice-to-Volume registrations: %s"
                  % (time_elapsed_tmp))

            # Super-resolution reconstruction
            time_elapsed_tmp = ph.start_timing()
            ph.print_subtitle("Cycle %d/%d: Super-Resolution Reconstruction"
                              % (i_cycle+1, args.two_step_cycles))

            SRR.set_alpha(args.alpha + i_cycle*alpha_delta)
            SRR.run_reconstruction()

            time_elapsed_tmp = ph.stop_timing(time_elapsed_tmp)
            SRR.print_statistics()
            time_reconstruction = ph.add_times(
                time_reconstruction, time_elapsed_tmp)

            HR_volume_tmp = HR_volume.get_stack_multiplied_with_mask(
                filename="HRvolume_iter"+str(i_cycle+1))
            HR_volume_iterations.insert(0, HR_volume_tmp)
            if args.verbose:
                sitkh.show_stacks(HR_volume_iterations)

    SRR.set_alpha(args.alpha_final)
    SRR.set_iter_max(args.iter_max_final)
    ph.print_subtitle("Final Super-Resolution Reconstruction")
    time_elapsed_tmp = ph.start_timing()
    SRR.run_reconstruction()
    time_elapsed_tmp = ph.stop_timing(time_elapsed_tmp)
    time_reconstruction = ph.add_times(time_reconstruction, time_elapsed_tmp)
    SRR.print_statistics()

    elapsed_time_total = ph.stop_timing(time_start)

    HR_volume_final = SRR.get_reconstruction().get_stack_multiplied_with_mask()
    HR_volume_final.set_filename(SRR.get_setting_specific_filename())
    HR_volume_final.write(args.dir_output)

    for stack in stacks:
        stack.write(
            os.path.join(args.dir_output,"motion_correction"),
            write_mask=True,
            write_slices=True,
            write_transforms=True,
            )

    HR_volume_iterations.insert(0, HR_volume_final)
    if args.verbose and not args.provide_comparison:
        sitkh.show_stacks(HR_volume_iterations)
    # HR_volume_final.show()

    # Show SRR together with linearly resampled input data.
    # Additionally, a script is generated to open files
    if args.provide_comparison:
        sitkh.show_stacks(HR_volume_iterations,
                          show_comparison_file=args.provide_comparison,
                          dir_output=os.path.join(
                              args.dir_output, "comparison"),
                          )

    # Summary
    ph.print_title("Summary")
    print("Computational Time for Data Preprocessing: %s" %
          (time_data_preprocessing))
    print("Computational Time for Registrations: %s" %
          (time_registration))
    print("Computational Time for Reconstructions: %s" %
          (time_reconstruction))
    print("Computational Time for Entire Reconstruction Pipeline: %s" %
          (elapsed_time_total))

    ph.print_line_separator()
