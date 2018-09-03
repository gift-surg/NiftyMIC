##
# \file reconstruct_volume_from_slices.py
# \brief      Script to reconstruct an isotropic, high-resolution volume from
#             multiple motion-corrected (or static) stacks of low-resolution 2D
#             slices.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       March 2017
#

# Import libraries
import numpy as np
import os

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.stack as st
import niftymic.base.data_reader as dr
import niftymic.reconstruction.admm_solver as admm
import niftymic.utilities.intensity_correction as ic
import niftymic.reconstruction.primal_dual_solver as pd
import niftymic.reconstruction.tikhonov_solver as tk
from niftymic.utilities.input_arparser import InputArgparser


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
    input_parser.add_dir_input_mc()
    input_parser.add_filenames(required=True)
    input_parser.add_filenames_masks()
    input_parser.add_dir_output(required=True)
    input_parser.add_prefix_output(default="SRR_")
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_target_stack_index(default=0)
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
    input_parser.add_rho(default=0.5)
    input_parser.add_tv_solver(default="PD")
    input_parser.add_pd_alg_type(default="ALG2")
    input_parser.add_iterations(default=15)
    input_parser.add_subfolder_comparison()
    input_parser.add_provide_comparison(default=0)
    input_parser.add_log_config(default=1)
    input_parser.add_use_masks_srr(default=0)
    input_parser.add_verbose(default=0)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")

    data_reader = dr.MultipleImagesReader(
        file_paths=args.filenames,
        file_paths_masks=args.filenames_masks,
        suffix_mask=args.suffix_mask,
        dir_motion_correction=args.dir_input_mc,
    )

    if args.reconstruction_type not in ["TK1L2", "TVL2", "HuberL2"]:
        raise IOError("Reconstruction type unknown")

    data_reader.read_data()
    stacks = data_reader.get_data()
    ph.print_info("%d input stacks read for further processing" % len(stacks))

    # ---------------------------Intensity Correction--------------------------
    if args.intensity_correction:
        ph.print_title("Intensity Correction")
        intensity_corrector = ic.IntensityCorrection()
        intensity_corrector.use_individual_slice_correction(False)
        intensity_corrector.use_stack_mask(True)
        intensity_corrector.use_reference_mask(True)
        intensity_corrector.use_verbose(False)

        for i, stack in enumerate(stacks):
            if i == args.target_stack_index:
                ph.print_info("Stack %d: Reference image. Skipped." % (i + 1))
                continue
            else:
                ph.print_info("Stack %d: Intensity Correction ... " % (i + 1),
                              newline=False)
            intensity_corrector.set_stack(stack)
            intensity_corrector.set_reference(
                stacks[args.target_stack_index].get_resampled_stack(
                    resampling_grid=stack.sitk,
                    interpolator="NearestNeighbor",
                ))
            intensity_corrector.run_linear_intensity_correction()
            stacks[i] = intensity_corrector.get_intensity_corrected_stack()
            print("done (c1 = %g) " %
                  intensity_corrector.get_intensity_correction_coefficients())

    # Reconstruction space is given isotropically resampled target stack
    if args.reconstruction_space is None:
        recon0 = \
            stacks[args.target_stack_index].get_isotropically_resampled_stack(
                resolution=args.isotropic_resolution,
                extra_frame=args.extra_frame_target)
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
        recon0 = \
            stacks[args.target_stack_index].get_resampled_stack(recon0.sitk)
        recon0 = recon0.get_stack_multiplied_with_mask()

    if args.reconstruction_type in ["TVL2", "HuberL2"]:
        ph.print_title("Compute Initial value for %s" %
                       args.reconstruction_type)
        SRR0 = tk.TikhonovSolver(
            stacks=stacks,
            reconstruction=recon0,
            alpha=args.alpha,
            iter_max=np.min([5, args.iter_max]),
            reg_type="TK1",
            minimizer="lsmr",
            data_loss="linear",
            use_masks=args.use_masks_srr,
            # verbose=args.verbose,
        )
    else:
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
    recon.set_filename(SRR0.get_setting_specific_filename(args.prefix_output))
    recon.write(args.dir_output)

    # List to store SRRs
    recons = []
    for i in range(0, len(stacks)):
        recons.append(stacks[i])
    recons.insert(0, recon)

    if args.reconstruction_type in ["TVL2", "HuberL2"]:
        ph.print_title("Compute %s reconstruction" % args.reconstruction_type)
        if args.tv_solver == "ADMM":
            SRR = admm.ADMMSolver(
                stacks=stacks,
                reconstruction=st.Stack.from_stack(SRR0.get_reconstruction()),
                minimizer=args.minimizer,
                alpha=args.alpha,
                iter_max=args.iter_max,
                rho=args.rho,
                data_loss=args.data_loss,
                iterations=args.iterations,
                use_masks=args.use_masks_srr,
                verbose=args.verbose,
            )
            SRR.run()
            recon = SRR.get_reconstruction()
            recon.set_filename(
                SRR.get_setting_specific_filename(args.prefix_output))
            recons.insert(0, recon)

            recon.write(args.dir_output)

        else:
            SRR = pd.PrimalDualSolver(
                stacks=stacks,
                reconstruction=st.Stack.from_stack(SRR0.get_reconstruction()),
                minimizer=args.minimizer,
                alpha=args.alpha,
                iter_max=args.iter_max,
                iterations=args.iterations,
                alg_type=args.pd_alg_type,
                reg_type="TV" if args.reconstruction_type == "TVL2" else "huber",
                data_loss=args.data_loss,
                verbose=args.verbose,
            )
            SRR.run()
            recon = SRR.get_reconstruction()
            recon.set_filename(
                SRR.get_setting_specific_filename(args.prefix_output))
            recons.insert(0, recon)

            recon.write(args.dir_output)

    if args.verbose and not args.provide_comparison:
        sitkh.show_stacks(recons)

    # Show SRR together with linearly resampled input data.
    # Additionally, a script is generated to open files
    if args.provide_comparison:
        sitkh.show_stacks(recons,
                          show_comparison_file=args.provide_comparison,
                          dir_output=os.path.join(
                              args.dir_output,
                              args.subfolder_comparison),
                          )

    ph.print_line_separator()

    elapsed_time = ph.stop_timing(time_start)
    ph.print_title("Summary")
    print("Computational Time for Volumetric Reconstruction: %s" %
          (elapsed_time))

    return 0


if __name__ == '__main__':
    main()
