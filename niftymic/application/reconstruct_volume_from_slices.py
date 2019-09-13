##
# \file reconstruct_volume_from_slices.py
# \brief      Script to reconstruct an isotropic, high-resolution volume from
#             multiple motion-corrected (or static) stacks of low-resolution 2D
#             slices.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       March 2017
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

from niftymic.definitions import ALLOWED_EXTENSIONS


def main():

    time_start = ph.start_timing()

    # Set print options for numpy
    np.set_printoptions(precision=3)

    # Read input
    input_parser = InputArgparser(
        description="Volumetric MRI reconstruction framework to reconstruct "
        "an isotropic, high-resolution 3D volume from multiple "
        "motion-corrected (or static) stacks of low-resolution slices.",
    )
    input_parser.add_filenames(required=True)
    input_parser.add_filenames_masks()
    input_parser.add_dir_input_mc()
    input_parser.add_output(required=True)
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_target_stack(default=None)
    input_parser.add_extra_frame_target(default=10)
    input_parser.add_isotropic_resolution(default=None)
    input_parser.add_intensity_correction(default=1)
    input_parser.add_reconstruction_space(default=None)
    input_parser.add_minimizer(default="lsmr")
    input_parser.add_iter_max(default=10)
    input_parser.add_reconstruction_type(default="TK1L2")
    input_parser.add_data_loss(default="linear")
    input_parser.add_data_loss_scale(default=1)
    input_parser.add_alpha(
        default=0.01  # TK1L2
        # default=0.006  #TVL2, HuberL2
    )
    input_parser.add_rho(default=0.1)
    input_parser.add_tv_solver(default="PD")
    input_parser.add_pd_alg_type(default="ALG2")
    input_parser.add_iterations(default=15)
    input_parser.add_log_config(default=1)
    input_parser.add_use_masks_srr(default=0)
    input_parser.add_slice_thicknesses(default=None)
    input_parser.add_verbose(default=0)
    input_parser.add_viewer(default="itksnap")
    input_parser.add_argument(
        "--mask", "-mask",
        action='store_true',
        help="If given, input images are interpreted as image masks. "
        "Obtained volumetric reconstruction will be exported in uint8 format."
    )
    input_parser.add_argument(
        "--sda", "-sda",
        action='store_true',
        help="If given, the volume is reconstructed using "
        "Scattered Data Approximation (Vercauteren et al., 2006). "
        "--alpha is considered the value for the standard deviation then. "
        "Recommended value is, e.g., --alpha 0.8"
    )

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    if args.reconstruction_type not in ["TK1L2", "TVL2", "HuberL2"]:
        raise IOError("Reconstruction type unknown")

    if np.alltrue([not args.output.endswith(t) for t in ALLOWED_EXTENSIONS]):
        raise ValueError(
            "output filename '%s' invalid; "
            "allowed image extensions are: %s" % (
                args.output, ", ".join(ALLOWED_EXTENSIONS)))

    dir_output = os.path.dirname(args.output)
    ph.create_directory(dir_output)

    debug = 0

    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))

    if args.verbose:
        show_niftis = []
        # show_niftis = [f for f in args.filenames]

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")

    if args.mask:
        filenames_masks = args.filenames
    else:
        filenames_masks = args.filenames_masks

    data_reader = dr.MultipleImagesReader(
        file_paths=args.filenames,
        file_paths_masks=filenames_masks,
        suffix_mask=args.suffix_mask,
        dir_motion_correction=args.dir_input_mc,
        stacks_slice_thicknesses=args.slice_thicknesses,
    )
    data_reader.read_data()
    stacks = data_reader.get_data()

    ph.print_info("%d input stacks read for further processing" % len(stacks))

    # Specify target stack for intensity correction and reconstruction space
    if args.target_stack is None:
        target_stack_index = 0
    else:
        # TODO: deal with case when target stack got rejected in previous step
        filenames = ["%s.nii.gz" % s.get_filename() for s in stacks]
        filename_target_stack = os.path.basename(args.target_stack)
        try:
            target_stack_index = filenames.index(filename_target_stack)
        except ValueError as e:
            raise ValueError(
                "--target-stack must correspond to an image as provided by "
                "--filenames")

    # ---------------------------Intensity Correction--------------------------
    if args.intensity_correction and not args.mask:
        ph.print_title("Intensity Correction")
        intensity_corrector = ic.IntensityCorrection()
        intensity_corrector.use_individual_slice_correction(False)
        intensity_corrector.use_stack_mask(True)
        intensity_corrector.use_reference_mask(True)
        intensity_corrector.use_verbose(False)

        for i, stack in enumerate(stacks):
            if i == target_stack_index:
                ph.print_info("Stack %d (%s): Reference image. Skipped." % (
                    i + 1, stack.get_filename()))
                continue
            else:
                ph.print_info("Stack %d (%s): Intensity Correction ... " % (
                    i + 1, stack.get_filename()), newline=False)
            intensity_corrector.set_stack(stack)
            intensity_corrector.set_reference(
                stacks[target_stack_index].get_resampled_stack(
                    resampling_grid=stack.sitk,
                    interpolator="NearestNeighbor",
                ))
            intensity_corrector.run_linear_intensity_correction()
            stacks[i] = intensity_corrector.get_intensity_corrected_stack()
            print("done (c1 = %g) " %
                  intensity_corrector.get_intensity_correction_coefficients())

    # -------------------------Volumetric Reconstruction-----------------------
    ph.print_title("Volumetric Reconstruction")

    # Reconstruction space defined by isotropically resampled,
    # bounding box-cropped target stack
    if args.reconstruction_space is None:
        recon0 = stacks[target_stack_index].get_isotropically_resampled_stack(
            resolution=args.isotropic_resolution,
            extra_frame=args.extra_frame_target,
        )
        recon0 = recon0.get_cropped_stack_based_on_mask(
            boundary_i=args.extra_frame_target,
            boundary_j=args.extra_frame_target,
            boundary_k=args.extra_frame_target,
            unit="mm",
        )

    # Reconstruction space was provided by user
    else:
        recon0 = st.Stack.from_filename(args.reconstruction_space,
                                        extract_slices=False)

        # Change resolution for isotropic resolution if provided by user
        if args.isotropic_resolution is not None:
            recon0 = recon0.get_isotropically_resampled_stack(
                args.isotropic_resolution)

        # Use image information of selected target stack as recon0 serves
        # as initial value for reconstruction
        recon0 = stacks[target_stack_index].get_resampled_stack(recon0.sitk)
        recon0 = recon0.get_stack_multiplied_with_mask()

    ph.print_info(
        "Reconstruction space defined with %s mm3 resolution" %
        " x ".join(["%.2f" % s for s in recon0.sitk.GetSpacing()])
    )

    if debug:
        # visualize (intensity corrected) data alongside recon0 init
        show = [st.Stack.from_stack(s) for s in stacks]
        show.insert(0, recon0)
        sitkh.show_stacks(show)

    if args.sda:
        ph.print_title("Compute SDA reconstruction")
        SDA = sda.ScatteredDataApproximation(
            stacks, recon0, sigma=args.alpha, sda_mask=args.mask)
        SDA.run()
        recon = SDA.get_reconstruction()
        filename = SDA.get_setting_specific_filename()
        if args.mask:
            dw.DataWriter.write_mask(
                recon.sitk_mask, args.output, description=filename)
        else:
            dw.DataWriter.write_image(
                recon.sitk, args.output, description=filename)

        if args.verbose:
            show_niftis.insert(0, args.output)

    else:
        if args.reconstruction_type in ["TVL2", "HuberL2"]:
            ph.print_title(
                "Compute Initial value for %s" % args.reconstruction_type)
            SRR0 = sda.ScatteredDataApproximation(stacks, recon0, sigma=0.8)
        else:
            ph.print_title(
                "Compute %s reconstruction" % args.reconstruction_type)
            SRR0 = tk.TikhonovSolver(
                stacks=stacks,
                reconstruction=recon0,
                alpha=args.alpha,
                iter_max=args.iter_max,
                reg_type="TK1",
                minimizer=args.minimizer,
                data_loss=args.data_loss,
                data_loss_scale=args.data_loss_scale,
                use_masks=args.use_masks_srr,
                # verbose=args.verbose,
            )
        SRR0.run()

        recon = SRR0.get_reconstruction()
        filename = SRR0.get_setting_specific_filename()

        if args.verbose and args.reconstruction_type in ["TVL2", "HuberL2"]:
            output = ph.append_to_filename(args.output, "_init")

            if args.mask:
                mask_estimator = bm.BinaryMaskFromMaskSRREstimator(recon.sitk)
                mask_estimator.run()
                mask_sitk = mask_estimator.get_mask_sitk()
                dw.DataWriter.write_mask(
                    mask_sitk, output, description=filename)
            else:
                dw.DataWriter.write_image(
                    recon.sitk, output, description=filename)

            show_niftis.insert(0, output)

        if args.reconstruction_type in ["TVL2", "HuberL2"]:
            ph.print_title("Compute %s reconstruction" %
                           args.reconstruction_type)
            if args.tv_solver == "ADMM":
                SRR = admm.ADMMSolver(
                    stacks=stacks,
                    reconstruction=st.Stack.from_stack(
                        SRR0.get_reconstruction()),
                    minimizer=args.minimizer,
                    alpha=args.alpha,
                    iter_max=args.iter_max,
                    rho=args.rho,
                    data_loss=args.data_loss,
                    iterations=args.iterations,
                    use_masks=args.use_masks_srr,
                    verbose=args.verbose,
                )

            else:
                SRR = pd.PrimalDualSolver(
                    stacks=stacks,
                    reconstruction=st.Stack.from_stack(
                        SRR0.get_reconstruction()),
                    minimizer=args.minimizer,
                    alpha=args.alpha,
                    iter_max=args.iter_max,
                    iterations=args.iterations,
                    alg_type=args.pd_alg_type,
                    reg_type="TV" if args.reconstruction_type == "TVL2" else "huber",
                    data_loss=args.data_loss,
                    use_masks=args.use_masks_srr,
                    verbose=args.verbose,
                )
            SRR.run()
            recon = SRR.get_reconstruction()
            filename = SRR.get_setting_specific_filename()

        if args.mask:
            mask_estimator = bm.BinaryMaskFromMaskSRREstimator(recon.sitk)
            mask_estimator.run()
            mask_sitk = mask_estimator.get_mask_sitk()
            dw.DataWriter.write_mask(
                mask_sitk, args.output, description=filename)

        else:
            dw.DataWriter.write_image(
                recon.sitk, args.output, description=filename)

        if args.verbose:
            show_niftis.insert(0, args.output)

    if args.verbose:
        ph.show_niftis(show_niftis, viewer=args.viewer)

    ph.print_line_separator()

    elapsed_time = ph.stop_timing(time_start)
    ph.print_title("Summary")
    exe_file_info = os.path.basename(os.path.abspath(__file__)).split(".")[0]
    print("%s | Computational Time for Volumetric Reconstruction: %s" % (
        exe_file_info, elapsed_time))

    return 0


if __name__ == '__main__':
    main()
