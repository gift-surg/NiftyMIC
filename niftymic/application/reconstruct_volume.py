##
# \file reconstruct_volume.py
# \brief      Script to reconstruct an isotropic, high-resolution volume from
#             multiple stacks of low-resolution 2D slices including
#             motion-correction.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       March 2017
#

import os
import numpy as np

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.stack as st
import niftymic.base.data_reader as dr
import niftymic.base.data_writer as dw
import niftymic.registration.flirt as regflirt
import niftymic.registration.niftyreg as niftyreg
import niftymic.registration.simple_itk_registration as regsitk
import niftymic.reconstruction.tikhonov_solver as tk
import niftymic.reconstruction.primal_dual_solver as pd
import niftymic.reconstruction.scattered_data_approximation as sda
import niftymic.utilities.data_preprocessing as dp
import niftymic.utilities.outlier_rejector as outre
import niftymic.utilities.intensity_correction as ic
import niftymic.utilities.joint_image_mask_builder as imb
import niftymic.utilities.segmentation_propagation as segprop
import niftymic.utilities.volumetric_reconstruction_pipeline as pipeline
from niftymic.utilities.input_arparser import InputArgparser

from niftymic.definitions import V2V_METHOD_OPTIONS, ALLOWED_EXTENSIONS


def main():

    time_start = ph.start_timing()

    # Set print options for numpy
    np.set_printoptions(precision=3)

    input_parser = InputArgparser(
        description="Volumetric MRI reconstruction framework to reconstruct "
        "an isotropic, high-resolution 3D volume from multiple stacks of 2D "
        "slices with motion correction. The resolution of the computed "
        "Super-Resolution Reconstruction (SRR) is given by the in-plane "
        "spacing of the selected target stack. A region of interest can be "
        "specified by providing a mask for the selected target stack. Only "
        "this region will then be reconstructed by the SRR algorithm which "
        "can substantially reduce the computational time.",
    )
    input_parser.add_filenames(required=True)
    input_parser.add_filenames_masks()
    input_parser.add_output(required=True)
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_target_stack(default=None)
    input_parser.add_search_angle(default=45)
    input_parser.add_multiresolution(default=0)
    input_parser.add_shrink_factors(default=[3, 2, 1])
    input_parser.add_smoothing_sigmas(default=[1.5, 1, 0])
    input_parser.add_sigma(default=1)
    input_parser.add_reconstruction_type(default="TK1L2")
    input_parser.add_iterations(default=15)
    input_parser.add_alpha(default=0.015)
    input_parser.add_alpha_first(default=0.2)
    input_parser.add_iter_max(default=10)
    input_parser.add_iter_max_first(default=5)
    input_parser.add_dilation_radius(default=3)
    input_parser.add_extra_frame_target(default=10)
    input_parser.add_bias_field_correction(default=0)
    input_parser.add_intensity_correction(default=1)
    input_parser.add_isotropic_resolution(default=1)
    input_parser.add_log_config(default=1)
    input_parser.add_subfolder_motion_correction()
    input_parser.add_write_motion_correction(default=1)
    input_parser.add_verbose(default=0)
    input_parser.add_two_step_cycles(default=3)
    input_parser.add_use_masks_srr(default=0)
    input_parser.add_boundary_stacks(default=[10, 10, 0])
    input_parser.add_metric(default="Correlation")
    input_parser.add_metric_radius(default=10)
    input_parser.add_reference()
    input_parser.add_reference_mask()
    input_parser.add_outlier_rejection(default=1)
    input_parser.add_threshold_first(default=0.5)
    input_parser.add_threshold(default=0.8)
    input_parser.add_use_robust_registration(default=0)
    input_parser.add_s2v_smoothing(default=0.5)
    input_parser.add_interleave(default=3)
    input_parser.add_slice_thicknesses(default=None)
    input_parser.add_viewer(default="itksnap")
    input_parser.add_v2v_method(default="RegAladin")
    input_parser.add_argument(
        "--v2v-robust", "-v2v-robust",
        action='store_true',
        help="If given, a more robust volume-to-volume registration step is "
        "performed, i.e. four rigid registrations are performed using four "
        "rigid transform initializations based on "
        "principal component alignment of associated masks."
    )
    input_parser.add_argument(
        "--s2v-hierarchical", "-s2v-hierarchical",
        action='store_true',
        help="If given, a hierarchical approach for the first slice-to-volume "
        "registration cycle is used, i.e. sub-packages defined by the "
        "specified interleave (--interleave) are registered until each "
        "slice is registered independently."
    )
    input_parser.add_argument(
        "--sda", "-sda",
        action='store_true',
        help="If given, the volumetric reconstructions are performed using "
        "Scattered Data Approximation (Vercauteren et al., 2006). "
        "'alpha' is considered the final 'sigma' for the "
        "iterative adjustment. "
        "Recommended value is, e.g., --alpha 0.8"
    )
    input_parser.add_option(
        option_string="--transforms-history",
        type=int,
        help="Write entire history of applied slice motion correction "
        "transformations to motion correction output directory",
        default=0,
    )

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    rejection_measure = "NCC"
    threshold_v2v = -2  # 0.3
    debug = False

    if args.v2v_method not in V2V_METHOD_OPTIONS:
        raise ValueError("v2v-method must be in {%s}" % (
            ", ".join(V2V_METHOD_OPTIONS)))

    if np.alltrue([not args.output.endswith(t) for t in ALLOWED_EXTENSIONS]):
        raise ValueError(
            "output filename invalid; allowed extensions are: %s" %
            ", ".join(ALLOWED_EXTENSIONS))

    if args.alpha_first < args.alpha and not args.sda:
        raise ValueError("It must hold alpha-first >= alpha")

    if args.threshold_first > args.threshold:
        raise ValueError("It must hold threshold-first <= threshold")

    dir_output = os.path.dirname(args.output)
    ph.create_directory(dir_output)

    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")
    data_reader = dr.MultipleImagesReader(
        file_paths=args.filenames,
        file_paths_masks=args.filenames_masks,
        suffix_mask=args.suffix_mask,
        stacks_slice_thicknesses=args.slice_thicknesses,
    )

    if len(args.boundary_stacks) is not 3:
        raise IOError(
            "Provide exactly three values for '--boundary-stacks' to define "
            "cropping in i-, j-, and k-dimension of the input stacks")

    data_reader.read_data()
    stacks = data_reader.get_data()
    ph.print_info("%d input stacks read for further processing" % len(stacks))

    if all(s.is_unity_mask() is True for s in stacks):
        ph.print_warning("No mask is provided! "
                         "Generated reconstruction space may be very big!")
        ph.print_warning("Consider using a mask to speed up computations")

        # args.extra_frame_target = 0
        # ph.wrint_warning("Overwritten: extra-frame-target set to 0")

    # Specify target stack for intensity correction and reconstruction space
    if args.target_stack is None:
        target_stack_index = 0
    else:
        try:
            target_stack_index = args.filenames.index(args.target_stack)
        except ValueError as e:
            raise ValueError(
                "--target-stack must correspond to an image as provided by "
                "--filenames")

    # ---------------------------Data Preprocessing---------------------------
    ph.print_title("Data Preprocessing")

    segmentation_propagator = segprop.SegmentationPropagation(
        # registration_method=regflirt.FLIRT(use_verbose=args.verbose),
        dilation_radius=args.dilation_radius,
        dilation_kernel="Ball",
    )

    data_preprocessing = dp.DataPreprocessing(
        stacks=stacks,
        segmentation_propagator=segmentation_propagator,
        use_cropping_to_mask=True,
        use_N4BiasFieldCorrector=args.bias_field_correction,
        target_stack_index=target_stack_index,
        boundary_i=args.boundary_stacks[0],
        boundary_j=args.boundary_stacks[1],
        boundary_k=args.boundary_stacks[2],
        unit="mm",
    )
    data_preprocessing.run()
    time_data_preprocessing = data_preprocessing.get_computational_time()

    # Get preprocessed stacks
    stacks = data_preprocessing.get_preprocessed_stacks()

    # Define reference/target stack for registration and reconstruction
    if args.reference is not None:
        reference = st.Stack.from_filename(
            file_path=args.reference,
            file_path_mask=args.reference_mask,
            extract_slices=False)

    else:
        reference = st.Stack.from_stack(stacks[target_stack_index])

    # ------------------------Volume-to-Volume Registration--------------------
    if len(stacks) > 1:

        if args.v2v_method == "FLIRT":
            # Define search angle ranges for FLIRT in all three dimensions
            search_angles = ["-searchr%s -%d %d" %
                             (x, args.search_angle, args.search_angle)
                             for x in ["x", "y", "z"]]
            options = (" ").join(search_angles)
            # options += " -noresample"

            vol_registration = regflirt.FLIRT(
                registration_type="Rigid",
                use_fixed_mask=True,
                use_moving_mask=True,
                options=options,
                use_verbose=False,
            )
        else:
            vol_registration = niftyreg.RegAladin(
                registration_type="Rigid",
                use_fixed_mask=True,
                use_moving_mask=True,
                # options="-ln 2 -voff",
                use_verbose=False,
            )
        v2vreg = pipeline.VolumeToVolumeRegistration(
            stacks=stacks,
            reference=reference,
            registration_method=vol_registration,
            verbose=debug,
            robust=args.v2v_robust,
        )
        v2vreg.run()
        stacks = v2vreg.get_stacks()
        time_registration = v2vreg.get_computational_time()

    else:
        time_registration = ph.get_zero_time()

    # ---------------------------Intensity Correction--------------------------
    if args.intensity_correction:
        ph.print_title("Intensity Correction")
        intensity_corrector = ic.IntensityCorrection()
        intensity_corrector.use_individual_slice_correction(False)
        intensity_corrector.use_reference_mask(True)
        intensity_corrector.use_stack_mask(True)
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

    # ---------------------------Create first volume---------------------------
    time_tmp = ph.start_timing()

    # Isotropic resampling to define HR target space
    ph.print_title("Reconstruction Space Generation")
    HR_volume = reference.get_isotropically_resampled_stack(
        resolution=args.isotropic_resolution)
    ph.print_info(
        "Isotropic reconstruction space with %g mm resolution is created" %
        HR_volume.sitk.GetSpacing()[0])

    if args.reference is None:
        # Create joint image mask in target space
        joint_image_mask_builder = imb.JointImageMaskBuilder(
            stacks=stacks,
            target=HR_volume,
            dilation_radius=1,
        )
        joint_image_mask_builder.run()
        HR_volume = joint_image_mask_builder.get_stack()
        ph.print_info(
            "Isotropic reconstruction space is centered around "
            "joint stack masks. ")

        # Crop to space defined by mask (plus extra margin)
        HR_volume = HR_volume.get_cropped_stack_based_on_mask(
            boundary_i=args.extra_frame_target,
            boundary_j=args.extra_frame_target,
            boundary_k=args.extra_frame_target,
            unit="mm",
        )

        # Create first volume
        # If outlier rejection is activated, eliminate obvious outliers early
        # from stack and re-run SDA to get initial volume without them
        ph.print_title("First Estimate of HR Volume")
        if args.outlier_rejection and threshold_v2v > -1:
            ph.print_subtitle("SDA Approximation")
            SDA = sda.ScatteredDataApproximation(
                stacks, HR_volume, sigma=args.sigma)
            SDA.run()
            HR_volume = SDA.get_reconstruction()

            # Identify and reject outliers
            ph.print_subtitle("Eliminate slice outliers (%s < %g)" % (
                rejection_measure, threshold_v2v))
            outlier_rejector = outre.OutlierRejector(
                stacks=stacks,
                reference=HR_volume,
                threshold=threshold_v2v,
                measure=rejection_measure,
                verbose=True,
            )
            outlier_rejector.run()
            stacks = outlier_rejector.get_stacks()

        ph.print_subtitle("SDA Approximation Image")
        SDA = sda.ScatteredDataApproximation(
            stacks, HR_volume, sigma=args.sigma)
        SDA.run()
        HR_volume = SDA.get_reconstruction()

        ph.print_subtitle("SDA Approximation Image Mask")
        SDA = sda.ScatteredDataApproximation(
            stacks, HR_volume, sigma=args.sigma, sda_mask=True)
        SDA.run()
        # HR volume contains updated mask based on SDA
        HR_volume = SDA.get_reconstruction()

        HR_volume.set_filename(SDA.get_setting_specific_filename())

    time_reconstruction = ph.stop_timing(time_tmp)

    if args.verbose:
        tmp = list(stacks)
        tmp.insert(0, HR_volume)
        sitkh.show_stacks(tmp, segmentation=HR_volume, viewer=args.viewer)

    # -----------Two-step Slice-to-Volume Registration-Reconstruction----------
    if args.two_step_cycles > 0:

        # Slice-to-volume registration set-up
        if args.metric == "ANTSNeighborhoodCorrelation":
            metric_params = {"radius": args.metric_radius}
        else:
            metric_params = None
        registration = regsitk.SimpleItkRegistration(
            moving=HR_volume,
            use_fixed_mask=True,
            use_moving_mask=True,
            interpolator="Linear",
            metric=args.metric,
            metric_params=metric_params,
            use_multiresolution_framework=args.multiresolution,
            shrink_factors=args.shrink_factors,
            smoothing_sigmas=args.smoothing_sigmas,
            initializer_type="SelfGEOMETRY",
            optimizer="ConjugateGradientLineSearch",
            optimizer_params={
                "learningRate": 1,
                "numberOfIterations": 100,
                "lineSearchUpperLimit": 2,
            },
            scales_estimator="Jacobian",
            use_verbose=debug,
        )

        # Volumetric reconstruction set-up
        if args.sda:
            recon_method = sda.ScatteredDataApproximation(
                stacks,
                HR_volume,
                sigma=args.sigma,
                use_masks=args.use_masks_srr,
            )
            alpha_range = [args.sigma, args.alpha]
        else:
            recon_method = tk.TikhonovSolver(
                stacks=stacks,
                reconstruction=HR_volume,
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
                stacks=stacks,
                reference=HR_volume,
                registration_method=registration,
                reconstruction_method=recon_method,
                cycles=args.two_step_cycles,
                alphas=alphas[0:args.two_step_cycles - 1],
                outlier_rejection=args.outlier_rejection,
                threshold_measure=rejection_measure,
                thresholds=thresholds,
                use_robust_registration=args.use_robust_registration,
                s2v_smoothing=args.s2v_smoothing,
                interleave=args.interleave,
                viewer=args.viewer,
                verbose=args.verbose,
                use_hierarchical_registration=args.s2v_hierarchical,
            )
        two_step_s2v_reg_recon.run()
        HR_volume_iterations = \
            two_step_s2v_reg_recon.get_iterative_reconstructions()
        time_registration += \
            two_step_s2v_reg_recon.get_computational_time_registration()
        time_reconstruction += \
            two_step_s2v_reg_recon.get_computational_time_reconstruction()
        stacks = two_step_s2v_reg_recon.get_stacks()

    # no two-step s2v-registration/reconstruction iterations
    else:
        HR_volume_iterations = []

    # Write motion-correction results
    ph.print_title("Write Motion Correction Results")
    if args.write_motion_correction:
        dir_output_mc = os.path.join(
            dir_output, args.subfolder_motion_correction)
        ph.clear_directory(dir_output_mc)

        for stack in stacks:
            stack.write(
                dir_output_mc,
                write_stack=False,
                write_mask=False,
                write_slices=False,
                write_transforms=True,
                write_transforms_history=args.transforms_history,
            )

        if args.outlier_rejection:
            deleted_slices_dic = {}
            for i, stack in enumerate(stacks):
                deleted_slices = stack.get_deleted_slice_numbers()
                deleted_slices_dic[stack.get_filename()] = deleted_slices

            # check whether any stack was removed entirely
            stacks0 = data_preprocessing.get_preprocessed_stacks()
            if len(stacks) != len(stacks0):
                stacks_remain = [s.get_filename() for s in stacks]
                for stack in stacks0:
                    if stack.get_filename() in stacks_remain:
                        continue

                    # add info that all slices of this stack were rejected
                    deleted_slices = [
                        slice.get_slice_number()
                        for slice in stack.get_slices()
                    ]
                    deleted_slices_dic[stack.get_filename()] = deleted_slices

            ph.write_dictionary_to_json(
                deleted_slices_dic,
                os.path.join(
                    dir_output,
                    args.subfolder_motion_correction,
                    "rejected_slices.json"
                )
            )

    # ---------------------Final Volumetric Reconstruction---------------------
    ph.print_title("Final Volumetric Reconstruction")
    if args.sda:
        recon_method = sda.ScatteredDataApproximation(
            stacks,
            HR_volume,
            sigma=args.alpha,
            use_masks=args.use_masks_srr,
        )
    else:
        if args.reconstruction_type in ["TVL2", "HuberL2"]:
            recon_method = pd.PrimalDualSolver(
                stacks=stacks,
                reconstruction=HR_volume,
                reg_type="TV" if args.reconstruction_type == "TVL2" else "huber",
                iterations=args.iterations,
                use_masks=args.use_masks_srr,
            )
        else:
            recon_method = tk.TikhonovSolver(
                stacks=stacks,
                reconstruction=HR_volume,
                reg_type="TK1" if args.reconstruction_type == "TK1L2" else "TK0",
                use_masks=args.use_masks_srr,
            )
        recon_method.set_alpha(args.alpha)
        recon_method.set_iter_max(args.iter_max)
        recon_method.set_verbose(True)
    recon_method.run()
    time_reconstruction += recon_method.get_computational_time()
    HR_volume_final = recon_method.get_reconstruction()

    ph.print_subtitle("Final SDA Approximation Image Mask")
    SDA = sda.ScatteredDataApproximation(
        stacks, HR_volume_final, sigma=args.sigma, sda_mask=True)
    SDA.run()
    # HR volume contains updated mask based on SDA
    HR_volume_final = SDA.get_reconstruction()
    time_reconstruction += SDA.get_computational_time()

    elapsed_time_total = ph.stop_timing(time_start)

    # Write SRR result
    filename = recon_method.get_setting_specific_filename()
    HR_volume_final.set_filename(filename)
    dw.DataWriter.write_image(
        HR_volume_final.sitk,
        args.output,
        description=filename)
    dw.DataWriter.write_mask(
        HR_volume_final.sitk_mask,
        ph.append_to_filename(args.output, "_mask"),
        description=SDA.get_setting_specific_filename())

    HR_volume_iterations.insert(0, HR_volume_final)
    for stack in stacks:
        HR_volume_iterations.append(stack)

    if args.verbose:
        sitkh.show_stacks(
            HR_volume_iterations,
            segmentation=HR_volume_final,
            viewer=args.viewer,
        )

    # Summary
    ph.print_title("Summary")
    exe_file_info = os.path.basename(os.path.abspath(__file__)).split(".")[0]
    print("%s | Computational Time for Data Preprocessing: %s" %
          (exe_file_info, time_data_preprocessing))
    print("%s | Computational Time for Registrations: %s" %
          (exe_file_info, time_registration))
    print("%s | Computational Time for Reconstructions: %s" %
          (exe_file_info, time_reconstruction))
    print("%s | Computational Time for Entire Reconstruction Pipeline: %s" %
          (exe_file_info, elapsed_time_total))

    ph.print_line_separator()

    return 0


if __name__ == '__main__':
    main()
