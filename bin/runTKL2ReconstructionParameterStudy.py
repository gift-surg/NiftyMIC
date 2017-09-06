#!/usr/bin/python

##
# \file studyTKL2ReconstructionParameters.py
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
import numericalsolver.Observer as Observer
import numericalsolver.TikhonovLinearSolverParameterStudy as tkparam
from numericalsolver.SimilarityMeasures import SimilarityMeasures as sim_meas

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
    input_parser.add_reference(required=True)
    input_parser.add_study_name()
    input_parser.add_dir_output(required=True)
    input_parser.add_image_selection()
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_minimizer(default="lsmr")
    input_parser.add_iter_max(default=10)
    input_parser.add_reg_type(default="TK1")
    input_parser.add_log_script_execution(default=1)
    input_parser.add_verbose(default=1)

    input_parser.add_alpha_range(default=[0.01, 0.05, 0.01])
    input_parser.add_data_losses(
        # default=["linear", "arctan"]
    )
    input_parser.add_data_loss_scale_range(
        # default=[0.1, 1.5, 0.5]
    )

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

    data_reader = dr.MultipleImagesReader(
        [args.reference],
        suffix_mask=args.suffix_mask,
        extract_slices=False)
    data_reader.read_data()
    reference = data_reader.get_stacks()[0]

    # ------------------------------Set Up Solver------------------------------
    reconstruction_space = stacks[0].get_resampled_stack(reference.sitk)

    SRR = tk.TikhonovSolver(
        stacks=stacks,
        reconstruction=reconstruction_space,
        iter_max=args.iter_max,
        reg_type=args.reg_type,
        minimizer=args.minimizer,
        verbose=args.verbose,
    )
    solver = SRR.get_solver()

    # ----------------------------Set Up Parameters----------------------------
    parameters = {}
    parameters["alpha"] = np.arange(*args.alpha_range)
    if args.data_losses is not None:
        parameters["data_loss"] = args.data_losses
    if args.data_loss_scale_range is not None:
        parameters["data_loss_scale"] = np.arange(*args.data_loss_scale_range)

    # -----------------------------Set Up Observer-----------------------------
    measures_dic = {}

    # Set up data cost
    A = SRR.get_A()
    b = SRR.get_b()
    measures_dic["Data_L2"] = lambda x: np.sum(np.square(A(x) - b))

    # Set up regularizer/prior cost
    B = solver.get_B()
    measures_dic["Reg_%s" % args.reg_type] = lambda x: np.sum(np.square(B(x)))

    # Set up similarity measures
    x_ref = sitk.GetArrayFromImage(reference.sitk).flatten()
    x_ref_mask = sitk.GetArrayFromImage(reference.sitk_mask).flatten()

    # Only evaluate similarity on mask
    indices = np.where(x_ref_mask > 0)

    measures_dic["SSD"] = \
        lambda x: sim_meas.sum_of_squared_differences(
            x[indices], x_ref[indices])
    measures_dic["MSE"] = \
        lambda x: sim_meas.mean_squared_error(
            x[indices], x_ref[indices])
    measures_dic["RMSE"] = \
        lambda x: sim_meas.root_mean_square_error(
            x[indices], x_ref[indices])
    measures_dic["PSNR"] = \
        lambda x: sim_meas.peak_signal_to_noise_ratio(
            x[indices], x_ref[indices])
    measures_dic["SSIM"] = \
        lambda x: sim_meas.structural_similarity(
            x[indices], x_ref[indices])
    measures_dic["NCC"] = \
        lambda x: sim_meas.normalized_cross_correlation(
            x[indices], x_ref[indices])
    measures_dic["MI"] = \
        lambda x: sim_meas.mutual_information(
            x[indices], x_ref[indices])
    measures_dic["NMI"] = \
        lambda x: sim_meas.normalized_mutual_information(
            x[indices], x_ref[indices])

    # Set up observer
    observer = Observer.Observer()
    observer.set_measures(measures_dic)

    # -------------------------Set Up Parameter Study-------------------------
    if args.study_name is None:
        name = args.reg_type + "L2"
    else:
        name = args.study_name

    parameter_study = tkparam.TikhonovLinearSolverParameterStudy(
        solver, observer,
        dir_output=args.dir_output,
        parameters=parameters,
        name=name,
    )

    # Run parameter study
    parameter_study.run()
