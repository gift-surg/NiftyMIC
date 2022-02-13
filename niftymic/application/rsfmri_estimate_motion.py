##
# \file rsfmri_estimate_motion.py
# \brief      Estimate motion in resting-state fMRI volumes
#
# \author     Michael Ebner (michael.ebner@kcl.ac.uk)
# \date       July 2019
#

import os
import numpy as np

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.stack as st
import niftymic.base.data_reader as dr
import niftymic.base.data_writer as dw
import niftymic.registration.flirt as regflirt
import niftymic.registration.niftyreg as regniftyreg
import niftymic.registration.simple_itk_registration as regsitk
import niftymic.reconstruction.tikhonov_solver as tk
import niftymic.reconstruction.primal_dual_solver as pd
import niftymic.reconstruction.scattered_data_approximation as sda
import niftymic.utilities.data_preprocessing as dp
import niftymic.utilities.joint_image_mask_builder as imb
import niftymic.utilities.segmentation_propagation as segprop
import niftymic.utilities.volumetric_reconstruction_pipeline as pipeline
from niftymic.utilities.input_arparser import InputArgparser

from niftymic.definitions import V2V_METHOD_OPTIONS


def main():

    time_start = ph.start_timing()

    # Set print options for numpy
    np.set_printoptions(precision=3)

    # Read input
    input_parser = InputArgparser(
        description="Estimate motion in resting-state fMRI volumes",
    )
    input_parser.add_filename(required=True)
    input_parser.add_filename_mask()
    input_parser.add_dir_output(required=True)
    input_parser.add_reference(required=False)
    input_parser.add_reference_mask(required=False)
    input_parser.add_alpha(default=0.03)
    input_parser.add_alpha_first(default=0.05)
    input_parser.add_data_loss(default="linear")
    input_parser.add_dilation_radius(default=3)
    input_parser.add_extra_frame_target(default=5)
    input_parser.add_isotropic_resolution(default=1)
    input_parser.add_iter_max(default=10)
    input_parser.add_iter_max_first(default=5)
    input_parser.add_iterations(default=10)
    input_parser.add_log_config(default=1)
    input_parser.add_minimizer(default="lsmr")
    input_parser.add_argument(
        "--prototyping", "-prototyping",
        action='store_true',
        help="If given, only a small subset of all time points is selected "
        "for quicker test computations."
    )
    input_parser.add_reconstruction_type(default="TK1L2")
    input_parser.add_rho(default=0.5)
    input_parser.add_sigma(default=0.8)
    input_parser.add_stack_recon_range(default=15)
    input_parser.add_target_stack_index(default=0)
    input_parser.add_two_step_cycles(default=3)
    input_parser.add_use_masks_srr(default=0)
    input_parser.add_verbose(default=0)
    input_parser.add_v2v_method(default="RegAladin")
    input_parser.add_outlier_rejection(default=1)
    input_parser.add_threshold_first(default=0.5)
    input_parser.add_threshold(default=0.8)
    input_parser.add_argument(
        "--sda", "-sda",
        action='store_true',
        help="If given, the volumetric reconstructions are performed using "
        "Scattered Data Approximation (Vercauteren et al., 2006). "
        "'alpha' is considered the final 'sigma' for the "
        "iterative adjustment. "
        "Recommended value is, e.g., --alpha 0.8"
    )

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    if args.v2v_method not in V2V_METHOD_OPTIONS:
        raise ValueError("v2v-method must be in {%s}" % (
            ", ".join(V2V_METHOD_OPTIONS)))

    if args.alpha_first < args.alpha and not args.sda:
        raise ValueError("It must hold alpha-first >= alpha")

    if args.threshold_first > args.threshold:
        raise ValueError("It must hold threshold-first <= threshold")

    # Write script execution call
    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")
    data_reader = dr.MultiComponentImageReader(
        args.filename, args.filename_mask)
    data_reader.read_data()
    stacks = data_reader.get_data()

    # ------------------------------DELETE LATER------------------------------
    if args.prototyping:
        stacks = stacks[0:2]
    # ------------------------------DELETE LATER------------------------------

    # ---------------------------Data Preprocessing---------------------------
    ph.print_title("Data Preprocessing")

    segmentation_propagator = segprop.SegmentationPropagation(
        # registration_method=regniftyreg.RegAladin(use_verbose=args.verbose),
        # registration_method=regsitk.SimpleItkRegistration(use_verbose=args.verbose),
        dilation_radius=args.dilation_radius,
        dilation_kernel="Ball",
    )

    data_preprocessing = dp.DataPreprocessing(
        stacks=stacks,
        segmentation_propagator=segmentation_propagator,

        # Not ideal: Entire FOV more desirable but registration is worse if not
        # cropped
        use_cropping_to_mask=args.use_masks_srr,

        target_stack_index=args.target_stack_index,
        boundary_i=0,
        boundary_j=0,
        boundary_k=0,
        unit="mm",
    )
    data_preprocessing.run()
    time_data_preprocessing = data_preprocessing.get_computational_time()

    # Get preprocessed stacks
    stacks = data_preprocessing.get_preprocessed_stacks()

    # Define volume-to-volume registration method
    if args.v2v_method == "FLIRT":
        registration_v2v = regflirt.FLIRT(
            registration_type="Rigid",
            use_fixed_mask=True,
            use_moving_mask=True,
            use_verbose=False,
        )
    else:
        registration_v2v = regniftyreg.RegAladin(
            registration_type="Rigid",
            use_fixed_mask=True,
            use_moving_mask=True,
            # options="-ln 2",
            use_verbose=False,
        )

    # Define slice-to-volume registration method
    registration_s2v = regsitk.SimpleItkRegistration(
        use_fixed_mask=True,
        use_moving_mask=True,
        use_verbose=False,
        interpolator="Linear",
        metric="Correlation",
        # metric="MattesMutualInformation",  # Might cause error messages
        # like "Too many samples map outside moving image buffer."
        # use_multiresolution_framework=True,
        shrink_factors=[2, 1],
        smoothing_sigmas=[1, 0],
        initializer_type="SelfGEOMETRY",
        optimizer="ConjugateGradientLineSearch",
        optimizer_params={
            "learningRate": 1,
            "numberOfIterations": 100,
            "lineSearchUpperLimit": 2,
        },
        # optimizer="RegularStepGradientDescent",
        # optimizer_params={
        #     "minStep": 1e-6,
        #     "numberOfIterations": 200,
        #     "gradientMagnitudeTolerance": 1e-6,
        #     "learningRate": 1,
        # },
        scales_estimator="Jacobian",
    )

    if args.reference is None:
        time_ref_estimate_start = ph.start_timing()

        # Stacks used for outlier-robust SRR algorithm
        i_min = args.target_stack_index
        i_max = np.min([args.target_stack_index + args.stack_recon_range,
                        len(stacks)])
        stacks_srr = [st.Stack.from_stack(s) for s in stacks[i_min: i_max]]

        # ----------------------Volume-to-Volume Registration------------------
        if args.two_step_cycles > 0:

            v2vreg = pipeline.VolumeToVolumeRegistration(
                stacks=stacks_srr,
                reference=stacks_srr[0],
                registration_method=registration_v2v,
                verbose=args.verbose,
            )
            v2vreg.run()
            stacks_srr = v2vreg.get_stacks()
            time_registration = v2vreg.get_computational_time()

        else:
            time_registration = ph.get_zero_time()

        # ---------------------------Create first volume-----------------------
        time_tmp = ph.start_timing()
        # Isotropic resampling to define HR target space
        ph.print_title("Isotropic Resampling")
        reference = stacks_srr[0].get_isotropically_resampled_stack(
            resolution=args.isotropic_resolution,
            extra_frame=args.extra_frame_target)

        # Scattered Data Approximation to get first estimate of HR volume
        ph.print_title("Scattered Data Approximation")
        SDA = sda.ScatteredDataApproximation(
            stacks=stacks_srr,
            HR_volume=reference,
            sigma=args.sigma,
            use_masks=args.use_masks_srr,
        )
        SDA.run()
        reference = SDA.get_reconstruction()

        joint_image_mask_builder = imb.JointImageMaskBuilder(
            stacks=stacks_srr,
            target=reference,
            dilation_radius=1,
        )
        joint_image_mask_builder.run()
        reference = joint_image_mask_builder.get_stack()
        reference.set_filename(SDA.get_setting_specific_filename())

        # Crop to space defined by mask (plus extra margin)
        reference = reference.get_cropped_stack_based_on_mask(
            boundary_i=args.extra_frame_target,
            boundary_j=args.extra_frame_target,
            boundary_k=args.extra_frame_target,
            unit="mm",
        )

        time_reconstruction = ph.stop_timing(time_tmp)

        # ----------------Two-step Slice-to-Volume Registration SRR------------
        if args.two_step_cycles > 0:

            # Volumetric reconstruction set-up
            if args.sda:
                recon_method = sda.ScatteredDataApproximation(
                    stacks=stacks_srr,
                    HR_volume=reference,
                    sigma=args.sigma,
                    use_masks=args.use_masks_srr,
                )
                alpha_range = [args.sigma, args.alpha]
            else:
                recon_method = tk.TikhonovSolver(
                    stacks=stacks_srr,
                    reconstruction=reference,
                    reg_type="TK1",
                    minimizer="lsmr",
                    alpha=args.alpha_first,
                    iter_max=np.min([args.iter_max_first, args.iter_max]),
                    verbose=True,
                    use_masks=args.use_masks_srr,
                )
                alpha_range = [args.alpha_first, args.alpha]

            # Define the regularization parameters for the individual
            # reconstruction steps in the two-step cycles
            alphas = np.linspace(
                alpha_range[0], alpha_range[1], args.two_step_cycles)

            # Define outlier rejection threshold after each S2V-reg step
            thresholds = np.linspace(
                args.threshold_first, args.threshold, args.two_step_cycles)

            two_step_s2v_reg_recon = \
                pipeline.TwoStepSliceToVolumeRegistrationReconstruction(
                    stacks=stacks_srr,
                    reference=reference,
                    registration_method=registration_s2v,
                    reconstruction_method=recon_method,
                    cycles=args.two_step_cycles,
                    alphas=alphas[0:args.two_step_cycles - 1],
                    verbose=args.verbose,
                    outlier_rejection=args.outlier_rejection,
                    thresholds=thresholds,
                )
            two_step_s2v_reg_recon.run()
            reference_iterations = \
                two_step_s2v_reg_recon.get_iterative_reconstructions()
            time_registration += \
                two_step_s2v_reg_recon.get_computational_time_registration()
            time_reconstruction += \
                two_step_s2v_reg_recon.get_computational_time_reconstruction()

            if args.verbose:
                sitkh.show_stacks(reference_iterations, segmentation=reference)

            # # Write to output
            # HR_volume_tmp.write(args.dir_output)

        ph.print_title("Final Reference Reconstruction")
        if args.sda:
            recon_method = sda.ScatteredDataApproximation(
                stacks_srr,
                reference,
                sigma=args.alpha,
                use_masks=args.use_masks_srr,
            )
        else:
            if args.reconstruction_type in ["TVL2", "HuberL2"]:
                recon_method = pd.PrimalDualSolver(
                    stacks=stacks_srr,
                    reconstruction=reference,
                    reg_type="TV" if args.reconstruction_type == "TVL2" else "huber",
                    iterations=args.iterations,
                    use_masks=args.use_masks_srr,
                )
            else:
                recon_method = tk.TikhonovSolver(
                    stacks=stacks_srr,
                    reconstruction=reference,
                    reg_type="TK1" if args.reconstruction_type == "TK1L2" else "TK0",
                    use_masks=args.use_masks_srr,
                )
            recon_method.set_alpha(args.alpha)
            recon_method.set_iter_max(args.iter_max)
            recon_method.set_verbose(True)
        recon_method.run()
        reference = recon_method.get_reconstruction()
        time_reconstruction += recon_method.get_computational_time()

        ph.print_subtitle("Final SDA Approximation Image Mask")
        SDA = sda.ScatteredDataApproximation(
            stacks_srr, reference, sigma=args.sigma, sda_mask=True)
        SDA.run()
        # Reference contains updated mask based on SDA
        reference = SDA.get_reconstruction()
        time_reconstruction += SDA.get_computational_time()

        description = recon_method.get_setting_specific_filename()
        reference.set_filename(description)
        name = "SDA" if args.sda else "SRR"
        path_to_reference = os.path.join(
            args.dir_output, "%s_reference.nii.gz" % name)
        dw.DataWriter.write_image(
            image_sitk=reference.sitk,
            path_to_file=path_to_reference,
            description=description)
        dw.DataWriter.write_mask(
            reference.sitk_mask,
            ph.append_to_filename(path_to_reference, "_mask"),
            description=SDA.get_setting_specific_filename())
        time_ref_estimate = ph.stop_timing(time_ref_estimate_start)

    else:
        reference = st.Stack.from_filename(args.reference, args.reference_mask)
        time_ref_estimate = ph.get_zero_time()
        ph.print_info("Reference image for V2V and S2V registrations provided")

    # --------------------Volume-to-Volume Registrations-----------------
    v2vreg = pipeline.VolumeToVolumeRegistration(
        stacks=stacks,
        reference=reference,
        registration_method=registration_v2v,
        verbose=False,
    )
    v2vreg.run()
    time_v2v_reg = v2vreg.get_computational_time()

    # --------------------Slice-to-Volume Registrations-----------------
    s2vreg = pipeline.SliceToVolumeRegistration(
        stacks=stacks,
        reference=reference,
        registration_method=registration_s2v,
        verbose=False,
    )
    s2vreg.run()
    time_s2v_reg = s2vreg.get_computational_time()

    # ------------------Write Slice Motion Correction Results------------------
    ph.print_title("Write Slice Motion Correction Results")
    dir_output_mc = os.path.join(
        args.dir_output,
        "motion_correction",
    )
    ph.clear_directory(dir_output_mc)
    for stack in stacks:
        stack.write(
            dir_output_mc,
            write_stack=False,
            write_mask=False,
            write_slices=False,
            write_transforms=True,
        )

    elapsed_time_total = ph.stop_timing(time_start)

    # Summary
    ph.print_title("Summary")
    exe_file_info = os.path.basename(os.path.abspath(__file__)).split(".")[0]
    print("%s | Computational Time for Data Preprocessing: %s" % (
        exe_file_info, time_data_preprocessing))
    print("%s | Computational Time for Reference (Estimate): %s" % (
        exe_file_info, time_ref_estimate))
    print("%s | Computational Time for V2V-Registration: %s" % (
        exe_file_info, time_v2v_reg))
    print("%s | Computational Time for S2V-Registration: %s" % (
        exe_file_info, time_s2v_reg))
    print("%s | Computational Time for Pipeline: %s" % (
        exe_file_info, elapsed_time_total))

    ph.print_line_separator()

    return 0


if __name__ == '__main__':
    main()
