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
from volumetricreconstruction.utilities.InputArparser import InputArgparser


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
    input_parser = InputArgparser(
        description="Volumetric MRI reconstruction framework to reconstruct "
        "an isotropic, high-resolution 3D volume from multiple stacks of 2D "
        "slices WITH motion correction. The resolution of the computed "
        "Super-Resolution Reconstruction (SRR) is given by the in-plane "
        "spacing of the selected target stack. A region of interest can be "
        "specified by providing a mask for the selected target stack. Only "
        "this region will then be reconstructed by the SRR algorithm which "
        "can substantially reduce the computational time.",
        prog="python " + os.path.basename(__file__),
    )
    input_parser.add_dir_input()
    input_parser.add_dir_output(default="results/")
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_prefix_output(default="_SRR")
    input_parser.add_target_stack_index(default=0)
    input_parser.add_reg_type(default="TK1")
    input_parser.add_data_loss(default="linear")
    input_parser.add_alpha(default=0.01)
    input_parser.add_isotropic_resolution(default=1)
    input_parser.add_iter_max(default=10)
    input_parser.add_rho(default=0.5)
    input_parser.add_admm_iterations(default=10)
    input_parser.add_minimizer(default="lsmr")
    input_parser.add_extra_frame_target(default=10)
    input_parser.add_provide_comparison(default=1)
    input_parser.add_log_script_execution(default=1)
    input_parser.add_verbose(default=1)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

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
