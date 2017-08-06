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
import numpy as np
import sys
import os

import pythonhelper.PythonHelper as ph
import pythonhelper.SimpleITKHelper as sitkh

# Import modules
import volumetricreconstruction.base.DataReader as dr
import volumetricreconstruction.base.Stack as st
import volumetricreconstruction.reconstruction.solver.TikhonovSolver as tk
import volumetricreconstruction.reconstruction.solver.ADMMSolver as admm
import volumetricreconstruction.reconstruction.solver.PrimalDualSolver as pd


##
# Gets the parsed input line.
# \date       2017-05-18 20:09:23+0100
#
# \param      dir_output             The dir output
# \param      filenames              The filenames
# \param      prefix_output          The prefix output
# \param      suffix_mask            The suffix mask
# \param      target_stack_index     The target stack index
# \param      two_step_cycles        The two step cycles
# \param      sigma                  The sigma
# \param      regularization         The regularization
# \param      data_loss              The data_loss
# \param      alpha                  The alpha
# \param      alpha_final            The alpha final
# \param      iter_max               The iterator maximum
# \param      iter_max_final         The iterator maximum final
# \param      minimizer              The minimizer
# \param      rho                    The rho
# \param      ADMM_iterations        The admm iterations
# \param      dilation_radius        The dilation radius
# \param      extra_frame_target     The extra frame target
# \param      bias_field_correction  The bias field correction
# \param      intensity_correction   The intensity correction
# \param      provide_comparison     The provide comparison
# \param      isotropic_resolution   The isotropic resolution
# \param      log_script_execution   The log script execution
# \param      log_motion_corretion   The log motion corretion
# \param      verbose                The verbose
#
# \return     The parsed input line.
#
def get_parsed_input_line(
    dir_output,
    prefix_output,
    suffix_mask,
    target_stack_index,
    regularization,
    data_loss,
    alpha,
    iter_max,
    minimizer,
    rho,
    ADMM_iterations,
    extra_frame_target,
    isotropic_resolution,
    log_script_execution,
    provide_comparison,
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
                        required=True,
                        default="")
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
                        "'TK0', 'TK1' or 'TV' for zeroth/first order Tikhonov "
                        " or total variation regularization, respectively."
                        "I.e. "
                        "R(x) = ||x||_2^2 for 'TK0', "
                        "R(x) = ||Dx||_2^2 for 'TK1', "
                        "or "
                        "R(x) = ||Dx||_1 for 'TV'. "
                        "[default: %s]"
                        % (regularization),
                        default=regularization)
    parser.add_argument('--iter-max',
                        type=int,
                        help="Number of maximum iterations for the numerical "
                        "solver. [default: %s]" % (iter_max),
                        default=iter_max)
    parser.add_argument('--rho',
                        type=float,
                        help="Regularization parameter for augmented "
                        "Lagrangian term for ADMM to solve the SR "
                        "reconstruction problem in case TV regularization is "
                        "chosen. "
                        "[default: %g]" % (rho),
                        default=rho)
    parser.add_argument('--ADMM-iterations',
                        type=int,
                        help="Number of ADMM iterations. "
                        "[default: %g]" % (ADMM_iterations),
                        default=ADMM_iterations)
    parser.add_argument('--minimizer',
                        type=str,
                        help="Choice of minimizer used for the inverse "
                        "problem associated to the SRR. Possible choices are "
                        "'lsmr' or 'L-BFGS-B'. Note, in case of a chosen "
                        "non-linear data loss only 'L-BFGS-B' is viable."
                        " [default: %s]" % (minimizer),
                        default=minimizer)
    parser.add_argument('--data-loss',
                        type=str,
                        help="Loss function rho used for data term, i.e. "
                        "rho((y_k - A_k x)^2). Possible choices are 'linear', "
                        "'soft_l1' or 'huber'. [default: %s]" % (data_loss),
                        default=data_loss)
    parser.add_argument('--extra-frame-target',
                        type=float,
                        help="Extra frame in mm added to the increase the "
                        "target space for the SRR. [default: %s]"
                        % (extra_frame_target),
                        default=extra_frame_target)
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
                        "allowing for a visual comparison of all obtained "
                        "reconstructions. An additional script "
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
        ph.print_info("%s: " % (arg), newline=False)
        print(getattr(args, arg))

    return args

"""
Main Function
"""
if __name__ == '__main__':

    run_ADMM = 1
    run_PrimalDual = 1

    alpha_ADMM = 0.1
    iter_max_ADMM = 5
    iterations_ADMM = 20

    alpha_PD = 0.1
    iter_max_PD = 5
    iterations_PD = 20

    time_start = ph.start_timing()

    # Set print options for numpy
    np.set_printoptions(precision=3)

    # Read input
    args = get_parsed_input_line(
        dir_output="results_recons/",
        prefix_output="SRR_",
        suffix_mask="_mask",
        target_stack_index=0,
        regularization="TK1",
        data_loss="linear",
        alpha=0.01,
        isotropic_resolution=None,
        iter_max=10,
        rho=0.5,
        ADMM_iterations=10,
        minimizer="lsmr",
        extra_frame_target=10,
        provide_comparison=1,
        log_script_execution=1,
        verbose=1,
    )

    # Write script execution call
    if args.log_script_execution:
        performed_script_execution = ph.get_performed_script_execution(
            os.path.basename(__file__), args)
        ph.write_performed_script_execution_to_executable_file(
            performed_script_execution,
            os.path.join(args.dir_output,
                         "log_%s_script_execution.sh" % (
                             os.path.basename(__file__).split(".")[0])))

    # Read Data:
    ph.print_title("Read Data")

    data_reader = dr.ImageSlicesDirectoryReader(
        args.dir_input, suffix_mask=args.suffix_mask)

    data_reader.read_data()
    stacks = data_reader.get_stacks()

    # if args.verbose:
    #     sitkh.show_stacks(stacks, segmentation=stacks[0])

    recon0 = stacks[args.target_stack_index].get_isotropically_resampled_stack(
        spacing_new_scalar=args.isotropic_resolution,
        extra_frame=args.extra_frame_target)

    SRR0 = tk.TikhonovSolver(
        stacks=stacks,
        reconstruction=recon0,
        alpha=args.alpha,
        iter_max=args.iter_max,
        reg_type="TK1",
        minimizer=args.minimizer,
        data_loss=args.data_loss,
        verbose=args.verbose,
    )
    SRR0.run_reconstruction()
    SRR0.print_statistics()

    recon = SRR0.get_reconstruction()
    recon.set_filename(SRR0.get_setting_specific_filename())
    recon.write(args.dir_output)

    # List to store SRRs
    recons = []
    for i in range(0, len(stacks)):
        recons.append(stacks[i])
    recons.insert(0, recon)

    if args.verbose:
        sitkh.show_stacks(recons)

    # for alpha in [0.5, 1, 2, 5, 10]:
    for alpha in [0.01]:
        if run_ADMM:
            SRR = admm.ADMMSolver(
                stacks=stacks,
                reconstruction=st.Stack.from_stack(SRR0.get_reconstruction()),
                minimizer=args.minimizer,
                alpha=alpha,
                iter_max=iter_max_ADMM,
                # iter_max=args.iter_max,
                rho=args.rho,
                # data_loss=args.data_loss,
                # iterations=args.ADMM_iterations,
                iterations=iterations_ADMM,
                verbose=args.verbose,
            )
            SRR.run_reconstruction()
            SRR.print_statistics()
            recon = SRR.get_reconstruction()
            recon.set_filename(SRR.get_setting_specific_filename())
            recons.insert(0, recon)

            recon.write(args.dir_output)

            if args.verbose:
                sitkh.show_stacks(recons)

        if run_PrimalDual:
            SRR = pd.PrimalDualSolver(
                stacks=stacks,
                reconstruction=st.Stack.from_stack(SRR0.get_reconstruction()),
                minimizer=args.minimizer,
                alpha=alpha,
                iter_max=iter_max_PD,
                iterations=iterations_PD,
                # alg_type="AHMOD",
                # reg_type="TV",
                # reg_type="huber",
                data_loss=args.data_loss,
                verbose=args.verbose,
            )
            SRR.run_reconstruction()
            SRR.print_statistics()
            recon = SRR.get_reconstruction()
            recon.set_filename(SRR.get_setting_specific_filename())
            recons.insert(0, recon)

            recon.write(args.dir_output)

            if args.verbose:
                sitkh.show_stacks(recons)

    if args.verbose and not args.provide_comparison:
        sitkh.show_stacks(recons)

    # Show SRR together with linearly resampled input data.
    # Additionally, a script is generated to open files
    if args.provide_comparison:
        sitkh.show_stacks(recons,
                          show_comparison_file=args.provide_comparison,
                          dir_output=os.path.join(
                              args.dir_output, "comparison"),
                          )

    ph.print_line_separator()
