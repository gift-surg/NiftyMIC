##
# \file run_reconstruction_parameter_study.py
# \brief      Script to study reconstruction parameters and their impact on the
#             volumetric reconstruction quality.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       September 2017
#

# Import libraries
import SimpleITK as sitk
import numpy as np
import os

# Import modules
import niftymic.base.data_reader as dr
import niftymic.base.stack as st
import niftymic.reconstruction.tikhonov_solver as tk
import nsol.deconvolution_solver_parameter_study_interface as \
    deconv_interface
import pysitk.python_helper as ph
from niftymic.utilities.input_arparser import InputArgparser


def main():

    time_start = ph.start_timing()

    # Set print options for numpy
    np.set_printoptions(precision=3)

    # Read input
    input_parser = InputArgparser(
        description="Script to study reconstruction parameters and their "
        "impact on the volumetric reconstruction quality. "
        "This script can only be used to sweep through one single parameter, "
        "e.g. the regularization parameter 'alpha'. "
    )
    input_parser.add_filenames(required=True)
    input_parser.add_filenames_masks()
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_dir_input_mc()
    input_parser.add_dir_output(required=True)
    input_parser.add_reconstruction_space()
    input_parser.add_reference(
        help="Path to reference NIfTI image file. If given the volumetric "
        "reconstructed is performed in this physical space. "
        "Either a reconstruction space or a reference must be provided",
        required=False)
    input_parser.add_reference_mask(default=None)
    input_parser.add_study_name()
    input_parser.add_reconstruction_type(default="TK1L2")
    input_parser.add_measures(
        default=["PSNR", "MAE", "RMSE", "SSIM", "NCC", "NMI"])
    input_parser.add_tv_solver(default="PD")
    input_parser.add_iterations(default=50)
    input_parser.add_rho(default=0.1)
    input_parser.add_iter_max(default=10)
    input_parser.add_minimizer(default="lsmr")
    input_parser.add_log_config(default=1)
    input_parser.add_use_masks_srr(default=0)
    input_parser.add_verbose(default=1)
    input_parser.add_slice_thicknesses(default=None)
    input_parser.add_argument(
        "--append", "-append",
        action='store_true',
        help="If given, results are appended to previously executed parameter "
        "study with identical parameters and study name store in the output "
        "directory."
    )

    # Range for parameter sweeps
    input_parser.add_alphas(default=list(np.linspace(0.01, 0.5, 5)))
    input_parser.add_data_losses(
        default=["linear"]
        # default=["linear", "arctan"]
    )
    input_parser.add_data_loss_scales(
        default=[1]
        # default=[0.1, 0.5, 1.5]
    )

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    if args.reference is None and args.reconstruction_space is None:
        raise IOError("Either reference (--reference) or reconstruction space "
                      "(--reconstruction-space) must be provided.")

    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")

    data_reader = dr.MultipleImagesReader(
        file_paths=args.filenames,
        file_paths_masks=args.filenames_masks,
        suffix_mask=args.suffix_mask,
        dir_motion_correction=args.dir_input_mc,
        stacks_slice_thicknesses=args.slice_thicknesses,
    )

    data_reader.read_data()
    stacks = data_reader.get_data()
    ph.print_info("%d input stacks read for further processing" % len(stacks))

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
    parameters["alpha"] = args.alphas
    if len(args.data_losses) > 1:
        parameters["data_loss"] = args.data_losses
    if len(args.data_loss_scales) > 1:
        parameters["data_loss_scale"] = args.data_loss_scales

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
        alpha=args.alphas[0],
        iter_max=args.iter_max,
        data_loss=args.data_losses[0],
        data_loss_scale=args.data_loss_scales[0],
        reg_type="TK1",
        minimizer=args.minimizer,
        verbose=args.verbose,
        use_masks=args.use_masks_srr,
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
            L2=16. / reconstruction_space.sitk.GetSpacing()[0]**2,
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
            append=args.append,
        )
    parameter_study_interface.set_up_parameter_study()
    parameter_study = parameter_study_interface.get_parameter_study()

    # Run parameter study
    parameter_study.run()

    print("\nComputational time for Deconvolution Parameter Study %s: %s" %
          (name, parameter_study.get_computational_time()))

    return 0


if __name__ == '__main__':
    main()
