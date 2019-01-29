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

from niftymic.definitions import V2V_METHOD_OPTIONS


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
    input_parser.add_dir_output(required=True)
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_target_stack_index(default=0)
    input_parser.add_search_angle(default=45)
    input_parser.add_multiresolution(default=0)
    input_parser.add_shrink_factors(default=[3, 2, 1])
    input_parser.add_smoothing_sigmas(default=[1.5, 1, 0])
    input_parser.add_sigma(default=1)
    input_parser.add_reconstruction_type(default="TK1L2")
    input_parser.add_iterations(default=15)
    input_parser.add_alpha(default=0.015)
    input_parser.add_alpha_first(default=0.05)
    input_parser.add_iter_max(default=10)
    input_parser.add_iter_max_first(default=5)
    input_parser.add_dilation_radius(default=3)
    input_parser.add_extra_frame_target(default=10)
    input_parser.add_bias_field_correction(default=0)
    input_parser.add_intensity_correction(default=1)
    input_parser.add_isotropic_resolution(default=1)
    input_parser.add_log_config(default=1)
    input_parser.add_subfolder_motion_correction()
    input_parser.add_provide_comparison(default=0)
    input_parser.add_subfolder_comparison()
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
    input_parser.add_interleave(default=2)
    input_parser.add_slice_thicknesses(default=None)
    input_parser.add_viewer(default="itksnap")
    input_parser.add_v2v_method(default="RegAladin")

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    if args.v2v_method not in V2V_METHOD_OPTIONS:
        raise ValueError("v2v-method must be in {%s}" % (
            ", ".join(V2V_METHOD_OPTIONS)))

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
        target_stack_index=args.target_stack_index,
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
        reference = st.Stack.from_stack(stacks[args.target_stack_index])

    # ------------------------Volume-to-Volume Registration--------------------
    if args.two_step_cycles > 0 and len(stacks) > 1:

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
                # options="-ln 2",
                use_verbose=False,
            )
        v2vreg = pipeline.VolumeToVolumeRegistration(
            stacks=stacks,
            reference=reference,
            registration_method=vol_registration,
            verbose=args.verbose,
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
        for reject_outliers in range(0, args.outlier_rejection + 1):
            if reject_outliers:
                # Identify and reject outliers
                threshold = 0.4
                ph.print_info("Eliminate obvious slice outliers: %g @ NCC" % (
                    threshold))
                outlier_rejector = outre.OutlierRejector(
                    stacks=stacks,
                    reference=HR_volume,
                    threshold=threshold,
                    measure="NCC",
                    verbose=True,
                )
                outlier_rejector.run()
                stacks = outlier_rejector.get_stacks()

            # Scattered Data Approximation to get first estimate of HR volume
            SDA = sda.ScatteredDataApproximation(
                stacks, HR_volume, sigma=args.sigma)
            SDA.run()
            HR_volume = SDA.get_reconstruction()
            HR_volume.set_filename("%s_isoSDA" % reference.get_filename())

    time_reconstruction = ph.stop_timing(time_tmp)

    if args.verbose:
        tmp = list(stacks)
        tmp.insert(0, HR_volume)
        sitkh.show_stacks(tmp, segmentation=HR_volume, viewer=args.viewer)

    # ----------------Two-step Slice-to-Volume Registration SRR----------------
    SRR = tk.TikhonovSolver(
        stacks=stacks,
        reconstruction=HR_volume,
        reg_type="TK1",
        minimizer="lsmr",
        alpha=args.alpha_first,
        iter_max=np.min([args.iter_max_first, args.iter_max]),
        verbose=True,
        use_masks=args.use_masks_srr,
    )

    if args.two_step_cycles > 0:

        if args.metric == "ANTSNeighborhoodCorrelation":
            metric_params = {"radius": args.metric_radius}
        else:
            metric_params = None

        registration = regsitk.SimpleItkRegistration(
            moving=HR_volume,
            use_fixed_mask=True,
            use_moving_mask=True,
            use_verbose=args.verbose,
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
        )
        two_step_s2v_reg_recon = \
            pipeline.TwoStepSliceToVolumeRegistrationReconstruction(
                stacks=stacks,
                reference=HR_volume,
                registration_method=registration,
                reconstruction_method=SRR,
                cycles=args.two_step_cycles,
                alpha_range=[args.alpha_first, args.alpha],
                verbose=args.verbose,
                outlier_rejection=args.outlier_rejection,
                threshold_measure="NCC",
                threshold_range=[args.threshold_first, args.threshold],
                use_robust_registration=args.use_robust_registration,
                s2v_smoothing=args.s2v_smoothing,
                interleave=args.interleave,
                viewer=args.viewer,
            )
        two_step_s2v_reg_recon.run()
        HR_volume_iterations = \
            two_step_s2v_reg_recon.get_iterative_reconstructions()
        time_registration += \
            two_step_s2v_reg_recon.get_computational_time_registration()
        time_reconstruction += \
            two_step_s2v_reg_recon.get_computational_time_reconstruction()
        stacks = two_step_s2v_reg_recon.get_stacks()
    else:
        HR_volume_iterations = []

    # Write motion-correction results
    ph.print_title("Write Motion Correction Results")
    if args.write_motion_correction:
        dir_output_mc = os.path.join(
            args.dir_output, args.subfolder_motion_correction)
        ph.clear_directory(dir_output_mc)

        for stack in stacks:
            stack.write(
                dir_output_mc,
                write_stack=False,
                write_mask=False,
                write_slices=False,
                write_transforms=True,
            )

        if args.outlier_rejection:
            deleted_slices_dic = {}
            for i, stack in enumerate(stacks):
                deleted_slices = stack.get_deleted_slice_numbers()
                deleted_slices_dic[stack.get_filename()] = deleted_slices
            ph.write_dictionary_to_json(
                deleted_slices_dic,
                os.path.join(
                    args.dir_output,
                    args.subfolder_motion_correction,
                    "rejected_slices.json"
                )
            )

    # ------------------Final Super-Resolution Reconstruction------------------
    ph.print_title("Final Super-Resolution Reconstruction")
    if args.reconstruction_type in ["TVL2", "HuberL2"]:
        SRR = pd.PrimalDualSolver(
            stacks=stacks,
            reconstruction=HR_volume,
            reg_type="TV" if args.reconstruction_type == "TVL2" else "huber",
            iterations=args.iterations,
        )
    else:
        SRR = tk.TikhonovSolver(
            stacks=stacks,
            reconstruction=HR_volume,
            reg_type="TK1" if args.reconstruction_type == "TK1L2" else "TK0",
            use_masks=args.use_masks_srr,
        )
    SRR.set_alpha(args.alpha)
    SRR.set_iter_max(args.iter_max)
    SRR.set_verbose(True)
    SRR.run()
    time_reconstruction += SRR.get_computational_time()

    elapsed_time_total = ph.stop_timing(time_start)

    # Write SRR result
    HR_volume_final = SRR.get_reconstruction()
    HR_volume_final.set_filename(SRR.get_setting_specific_filename())
    HR_volume_final.write(
        args.dir_output, write_mask=True, suffix_mask=args.suffix_mask)

    HR_volume_iterations.insert(0, HR_volume_final)
    for stack in stacks:
        HR_volume_iterations.append(stack)

    if args.verbose and not args.provide_comparison:
        sitkh.show_stacks(
            HR_volume_iterations,
            segmentation=HR_volume,
            viewer=args.viewer)
    # HR_volume_final.show()

    # Show SRR together with linearly resampled input data.
    # Additionally, a script is generated to open files
    if args.provide_comparison:
        sitkh.show_stacks(HR_volume_iterations,
                          segmentation=HR_volume,
                          show_comparison_file=args.provide_comparison,
                          dir_output=os.path.join(
                              args.dir_output, args.subfolder_comparison),
                          )

    # Summary
    ph.print_title("Summary")
    print("Computational Time for Data Preprocessing: %s" %
          (time_data_preprocessing))
    print("Computational Time for Registrations: %s" %
          (time_registration))
    print("Computational Time for Reconstructions: %s" %
          (time_reconstruction))
    print("Computational Time for Entire Reconstruction Pipeline: %s" %
          (elapsed_time_total))

    ph.print_line_separator()

    return 0


if __name__ == '__main__':
    main()
