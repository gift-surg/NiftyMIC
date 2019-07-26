##
# \file reconstruct_volume_from_slices_rsfmri.py
# \brief      Script to reconstruct an isotropic, high-resolution volume from
#             multiple motion-corrected (or static) stacks of low-resolution 2D
#             slices.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2019
#

import os
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.stack as st
import niftymic.base.data_reader as dr
import niftymic.base.data_writer as dw
import niftymic.reconstruction.admm_solver as admm
import niftymic.utilities.intensity_correction as ic
import niftymic.reconstruction.primal_dual_solver as pd
import niftymic.reconstruction.tikhonov_solver as tk
import niftymic.reconstruction.scattered_data_approximation as sda
import niftymic.utilities.binary_mask_from_mask_srr_estimator as bm
from niftymic.utilities.input_arparser import InputArgparser
import niftymic.utilities.volumetric_reconstruction_pipeline as pipeline

from niftymic.definitions import ALLOWED_EXTENSIONS


def main():

    time_start = ph.start_timing()

    # Set print options for numpy
    np.set_printoptions(precision=3)

    # Read input
    input_parser = InputArgparser(
        description="Volumetric MRI reconstruction framework to reconstruct "
        "resting-state fMRI based on motion-corrected slice transformations "
        "obtained using reconstruct_volume_rsfmri.",
    )
    input_parser.add_filename(required=True)
    input_parser.add_filename_mask()
    input_parser.add_output(required=True)
    input_parser.add_dir_input_mc()
    input_parser.add_argument(
        "--volume", "-vol",
        action='store_true',
        help="If given, reconstructions for each time point are performed "
        "based on volumetric stack position update only. "
        "Otherwise, reconstructions are based after performed motion updates "
        "for each individual slice for each time point."
    )
    input_parser.add_reconstruction_space(default=None)
    input_parser.add_alpha(default=0.05)
    input_parser.add_reconstruction_type(default="TK1L2")
    input_parser.add_data_loss(default="linear")
    input_parser.add_minimizer(default="lsmr")
    input_parser.add_iter_max(default=10)
    input_parser.add_iterations(default=10)
    input_parser.add_argument(
        "--prototyping", "-prototyping",
        action='store_true',
        help="If given, only a small subset of all time points is selected "
        "for quicker test computations."
    )
    input_parser.add_option(
        option_string="--reconstruction-spacing",
        type=float,
        nargs="+",
        help="Specify spacing of reconstruction space in case a change is desired",
        default=None)
    input_parser.add_argument(
        "--sda", "-sda",
        action='store_true',
        help="If given, the volumetric reconstructions are performed using "
        "Scattered Data Approximation (Vercauteren et al., 2006). "
        "'alpha' is considered the final 'sigma' for the "
        "iterative adjustment. "
        "Recommended value is, e.g., --alpha 0.8"
    )
    input_parser.add_use_masks_srr(default=1)
    input_parser.add_log_config(default=1)
    input_parser.add_verbose(default=0)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # Write script execution call
    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")
    data_reader = dr.MultiComponentImageReader(
        path_to_image=args.filename,
        path_to_image_mask=args.filename_mask,
        dir_motion_correction=args.dir_input_mc,
        volume_motion_only=args.volume,
    )
    data_reader.read_data()
    stacks = data_reader.get_data()

    # Define reconstruction space for rsfmri
    if args.reconstruction_space is None:
        path_to_recon_space = args.filename
    else:
        path_to_recon_space = args.reconstruction_space
    image_sitk = sitk.ReadImage(path_to_recon_space)

    # Standard 3D image
    # Multi-component 3D image
    if len(image_sitk.GetSize()) == 4:
        # Extract first component (3D image) of the 4D image
        shape = list(image_sitk.GetSize())
        shape[-1] = 0
        index = [0] * 4
        recon_space_sitk = sitk.Extract(image_sitk, shape, index)
    elif len(image_sitk.GetSize()) == 3:
        recon_space_sitk = image_sitk
    else:
        raise ValueError("Provide either a multi-component or a standard "
                         "3D image to define the reconstruction space")

    reconstruction_space = st.Stack.from_sitk_image(
        recon_space_sitk * 0,
        slice_thickness=recon_space_sitk.GetSpacing()[-1],
        filename=ph.strip_filename_extension(
            os.path.basename(path_to_recon_space))[0],
    )

    if args.reconstruction_spacing is not None:
        reconstruction_space = reconstruction_space.get_resampled_stack(
            spacing=args.reconstruction_spacing)

    # ------------------------------DELETE LATER------------------------------
    if args.prototyping:
        stacks = stacks[0:2]
    # ------------------------------DELETE LATER------------------------------

    # ----Define solver for rsfMRI reconstructions of individual timepoints----
    if args.sda:
        recon_method = sda.ScatteredDataApproximation(
            stacks,
            reconstruction_space,
            sigma=args.alpha,
            use_masks=args.use_masks_srr,
        )
    else:
        if args.reconstruction_type in ["TVL2", "HuberL2"]:
            recon_method = pd.PrimalDualSolver(
                stacks=stacks,
                reconstruction=reconstruction_space,
                reg_type="TV" if args.reconstruction_type == "TVL2" else "huber",
                iterations=args.iterations,
                use_masks=args.use_masks_srr,
            )
        else:
            recon_method = tk.TikhonovSolver(
                stacks=stacks,
                reconstruction=reconstruction_space,
                reg_type="TK1" if args.reconstruction_type == "TK1L2" else "TK0",
                use_masks=args.use_masks_srr,
            )
        recon_method.set_alpha(args.alpha)
        recon_method.set_iter_max(args.iter_max)
        recon_method.set_verbose(True)

    # ------Update individual timepoints based on updated slice positions------
    multi_component_reconstruction = pipeline.MultiComponentReconstruction(
        stacks=stacks,
        reconstruction_method=recon_method,
        suffix="_recon_v2v")
    multi_component_reconstruction.run()
    time_reconstruction = \
        multi_component_reconstruction.get_computational_time()
    stacks_recon = multi_component_reconstruction.get_reconstructions()

    # --------------------------------Write Data------------------------------
    ph.print_title("Write Data")
    description = multi_component_reconstruction.get_reconstruction_method().\
        get_setting_specific_filename()
    data_writer = dw.MultiComponentImageWriter(
        stacks_recon, args.output, description=description)
    data_writer.write_data()

    if args.verbose:
        ph.show_nifti(args.output)

    elapsed_time = ph.stop_timing(time_start)

    ph.print_title("Summary")
    exe_file_info = os.path.basename(os.path.abspath(__file__)).split(".")[0]
    print("%s | Computational Time for Volumetric Reconstruction: %s" %
          (exe_file_info, elapsed_time))

    return 0


if __name__ == '__main__':
    main()
