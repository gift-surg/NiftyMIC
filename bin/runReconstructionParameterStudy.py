#!/usr/bin/python

##
# \file runReconstructionParameterStudy.py
# \brief      Script to study reconstruction parameters for least squares
#             reconstruction with Tikhonov regularization.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       September 2017
#

# Import libraries
import SimpleITK as sitk
import argparse
import numpy as np
import sys
import os

import pythonhelper.PythonHelper as ph
import pythonhelper.SimpleITKHelper as sitkh
import numericalsolver.DeconvolutionSolverParameterStudyInterface as \
    deconv_interface

# Import modules
import volumetricreconstruction.base.DataReader as dr
import volumetricreconstruction.base.Stack as st
import volumetricreconstruction.utilities.Exceptions as Exceptions
import volumetricreconstruction.reconstruction.solver.TikhonovSolver as tk
from volumetricreconstruction.utilities.InputArparser import InputArgparser


if __name__ == '__main__':

    time_start = ph.start_timing()

    # Set print options for numpy
    np.set_printoptions(precision=3)

    # Read input
    input_parser = InputArgparser(
        description="",
        prog="python " + os.path.basename(__file__),
    )
    input_parser.add_dir_input()
    input_parser.add_filenames()
    input_parser.add_image_selection()
    input_parser.add_dir_output(required=True)
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_reconstruction_space()
    input_parser.add_reference(
        help="Path to reference NIfTI image file. If given SRR is "
        "reconstructed in this physical space. "
        "Either a reconstruction space or a reference must be provided",
        required=False)
    input_parser.add_reference_mask(default=None)
    input_parser.add_study_name()
    input_parser.add_reconstruction_type(default="TK1L2")
    input_parser.add_measures(default=["PSNR", "RMSE", "SSIM", "NCC", "NMI"])
    input_parser.add_tv_solver(default="PD")
    input_parser.add_iterations(default=50)
    input_parser.add_rho(default=0.1)
    input_parser.add_iter_max(default=10)
    input_parser.add_minimizer(default="lsmr")
    input_parser.add_alpha(default=0.01)
    input_parser.add_data_loss(default="linear")
    input_parser.add_data_loss_scale(default=1)
    input_parser.add_log_script_execution(default=1)
    input_parser.add_verbose(default=1)

    # Range for parameter sweeps
    input_parser.add_alpha_range(default=[0.001, 0.05, 20])  # TK1L2
    # input_parser.add_alpha_range(default=[0.001, 0.003, 10])  # TVL2, HuberL2
    input_parser.add_data_losses(
        # default=["linear", "arctan"]
    )
    input_parser.add_data_loss_scale_range(
        # default=[0.1, 1.5, 2]
    )

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    if args.reference is None and args.reconstruction_space is None:
        raise IOError("Either reference (--reference) or reconstruction space "
                      "(--reconstruction-space) must be provided.")

    # Write script execution call
    if args.log_script_execution:
        input_parser.write_performed_script_execution(
            os.path.abspath(__file__))

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")

    # Neither '--dir-input' nor '--filenames' was specified
    if args.filenames is not None and args.dir_input is not None:
        raise Exceptions.IOError(
            "Provide input by either '--dir-input' or '--filenames' "
            "but not both together")

    # '--dir-input' specified
    elif args.dir_input is not None:
        data_reader = dr.ImageSlicesDirectoryReader(
            path_to_directory=args.dir_input,
            suffix_mask=args.suffix_mask,
            image_selection=args.image_selection)

    # '--filenames' specified
    elif args.filenames is not None:
        data_reader = dr.MultipleImagesReader(
            args.filenames, suffix_mask=args.suffix_mask)

    else:
        raise Exceptions.IOError(
            "Provide input by either '--dir-input' or '--filenames'")

    data_reader.read_data()
    stacks = data_reader.get_stacks()

    if args.reference is not None:
        reference = st.Stack.from_filename(
            file_path=args.reference,
            file_path_mask=args.reference_mask,
            extract_slices=False)

        reconstruction_space = stacks[0].get_resampled_stack(reference.sitk)
        reconstruction_space = \
            reconstruction_space.get_stack_multiplied_with_mask()
        x_ref = sitk.GetArrayFromImage(reference.sitk).flatten()
        x_ref_mask = sitk.GetArrayFromImage(reference.sitk_mask).flatten()

    else:
        reconstruction_space = st.Stack.from_filename(
            file_path=args.reconstruction_space,
            extract_slices=False)
        reconstruction_space = stacks[0].get_resampled_stack(
            reconstruction_space.sitk)
        reconstruction_space = \
            reconstruction_space.get_stack_multiplied_with_mask()
        x_ref = None
        x_ref_mask = None

    # ----------------------------Set Up Parameters----------------------------
    parameters = {}
    parameters["alpha"] = np.linspace(
        args.alpha_range[0], args.alpha_range[1], int(args.alpha_range[2]))
    if args.data_losses is not None:
        parameters["data_loss"] = args.data_losses
    if args.data_loss_scale_range is not None:
        parameters["data_loss_scale"] = np.linspace(
            args.data_loss_scale_range[0],
            args.data_loss_scale_range[1],
            int(args.data_loss_scale_range[2]))

    # --------------------------Set Up Parameter Study-------------------------
    if args.study_name is None:
        name = args.reconstruction_type
    else:
        name = args.study_name

    reconstruction_info = {
        "shape": reconstruction_space.sitk.GetSize()[::-1],
        "origin": reconstruction_space.sitk.GetOrigin(),
        "spacing": reconstruction_space.sitk.GetSpacing(),
        "direction": reconstruction_space.sitk.GetDirection(),
    }

    # Create Tikhonov solver from which all information can be extracted
    # (also for other reconstruction types)
    tmp = tk.TikhonovSolver(
        stacks=stacks,
        reconstruction=reconstruction_space,
        alpha=args.alpha,
        iter_max=args.iter_max,
        data_loss=args.data_loss,
        data_loss_scale=args.data_loss_scale,
        reg_type="TK1",
        minimizer=args.minimizer,
        verbose=args.verbose,
    )
    solver = tmp.get_solver()

    parameter_study_interface = \
        deconv_interface.DeconvolutionParameterStudyInterface(
            A=solver.get_A(),
            A_adj=solver.get_A_adj(),
            D=solver.get_B(),
            D_adj=solver.get_B_adj(),
            b=solver.get_b(),
            x0=solver.get_x0(),
            alpha=solver.get_alpha(),
            x_scale=solver.get_x_scale(),
            data_loss=solver.get_data_loss(),
            data_loss_scale=solver.get_data_loss_scale(),
            iter_max=solver.get_iter_max(),
            minimizer=solver.get_minimizer(),
            iterations=args.iterations,
            measures=args.measures,
            dimension=3,
            L2=16./reconstruction_space.sitk.GetSpacing()[0]**2,
            reconstruction_type=args.reconstruction_type,
            rho=args.rho,
            dir_output=args.dir_output,
            parameters=parameters,
            name=name,
            reconstruction_info=reconstruction_info,
            x_ref=x_ref,
            x_ref_mask=x_ref_mask,
            tv_solver=args.tv_solver,
            verbose=args.verbose,
        )
    parameter_study_interface.set_up_parameter_study()
    parameter_study = parameter_study_interface.get_parameter_study()

    # Run parameter study
    parameter_study.run()

    print("\nComputational time for Deconvolution Parameter Study %s: %s" %
          (name, parameter_study.get_computational_time()))
