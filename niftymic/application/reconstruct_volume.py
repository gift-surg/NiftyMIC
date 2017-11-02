##
# \file reconstruct_volume.py
# \brief      Script to reconstruct an isotropic, high-resolution volume from
#             multiple stacks of low-resolution 2D slices including
#             motion-correction.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       March 2017
#

# Import libraries
import os
import numpy as np

import niftymic.base.data_reader as dr
import niftymic.base.stack as st
import niftymic.reconstruction.primal_dual_solver as pd
import niftymic.reconstruction.scattered_data_approximation as \
    sda
import niftymic.reconstruction.tikhonov_solver as tk
import niftymic.registration.flirt as regflirt
import niftymic.registration.simple_itk_registration as regsitk
import niftymic.utilities.data_preprocessing as dp
import niftymic.utilities.segmentation_propagation as segprop
import niftymic.utilities.volumetric_reconstruction_pipeline as \
    pipeline
import niftymic.utilities.joint_image_mask_builder as imb
import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh
from niftymic.utilities.input_arparser import InputArgparser


def main():

    time_start = ph.start_timing()

    # Set print options for numpy
    np.set_printoptions(precision=3)

    # Read input
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
    input_parser.add_dir_input()
    input_parser.add_filenames()
    input_parser.add_dir_output(required=True)
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_target_stack_index(default=0)
    input_parser.add_search_angle(default=90)
    input_parser.add_multiresolution(default=0)
    input_parser.add_shrink_factors(default=[2, 1])
    input_parser.add_smoothing_sigmas(default=[1, 0])
    input_parser.add_sigma(default=0.9)
    input_parser.add_reconstruction_type(default="TK1L2")
    input_parser.add_iterations(default=15)
    input_parser.add_alpha(default=0.02)
    input_parser.add_alpha_first(default=0.05)
    input_parser.add_iter_max(default=10)
    input_parser.add_iter_max_first(default=5)
    input_parser.add_dilation_radius(default=3)
    input_parser.add_extra_frame_target(default=10)
    input_parser.add_bias_field_correction(default=0)
    input_parser.add_intensity_correction(default=0)
    input_parser.add_isotropic_resolution(default=None)
    input_parser.add_log_script_execution(default=1)
    input_parser.add_subfolder_motion_correction()
    input_parser.add_provide_comparison(default=0)
    input_parser.add_subfolder_comparison()
    input_parser.add_write_motion_correction(default=1)
    input_parser.add_verbose(default=0)
    input_parser.add_two_step_cycles(default=3)
    input_parser.add_reference()
    input_parser.add_reference_mask()

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # Write script execution call
    if args.log_script_execution:
        input_parser.write_performed_script_execution(
            os.path.abspath(__file__))

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")

    # Neither '--dir-input' nor '--filenames' was specified
    if args.filenames is not None and args.dir_input is not None:
        raise IOError(
            "Provide input by either '--dir-input' or '--filenames' "
            "but not both together")

    # '--dir-input' specified
    elif args.dir_input is not None:
        data_reader = dr.DirectoryReader(
            args.dir_input, suffix_mask=args.suffix_mask)

    # '--filenames' specified
    elif args.filenames is not None:
        data_reader = dr.MultipleImagesReader(
            args.filenames, suffix_mask=args.suffix_mask)

    else:
        raise IOError(
            "Provide input by either '--dir-input' or '--filenames'")

    data_reader.read_data()
    stacks = data_reader.get_stacks()

    if all(s.is_unity_mask() is True for s in stacks):
        ph.print_line_separator(symbol="X")
        ph.print_info(
            "WARNING: No mask is not provided! "
            "Generated reconstruction space may be very big!")
        ph.print_line_separator(symbol="X", add_newline=False)

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
        use_intensity_correction=args.intensity_correction,
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

    # Define reference/target stack for registration and reconstruction
    if args.reference is not None:
        reference = st.Stack.from_filename(
            file_path=args.reference,
            file_path_mask=args.reference_mask,
            extract_slices=False)
        if args.reference_mask is None:
            use_reference_mask = False
    else:
        reference = st.Stack.from_stack(stacks[args.target_stack_index])
        use_reference_mask = True

    if args.verbose:
        sitkh.show_stacks(stacks, segmentation=stacks[0])

    # ------------------------Volume-to-Volume Registration--------------------
    if args.two_step_cycles > 0:
        # Define search angle ranges for FLIRT in all three dimensions
        search_angles = ["-searchr%s -%d %d" %
                         (x, args.search_angle, args.search_angle)
                         for x in ["x", "y", "z"]]
        search_angles = (" ").join(search_angles)

        vol_registration = regflirt.FLIRT(
            registration_type="Rigid",
            use_fixed_mask=True,
            use_moving_mask=use_reference_mask,
            options=search_angles,
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

    # ---------------------------Create first volume---------------------------
    time_tmp = ph.start_timing()

    # Isotropic resampling to define HR target space
    ph.print_title("Reconstruction Space Generation")
    HR_volume = reference.get_isotropically_resampled_stack(
        spacing_new_scalar=args.isotropic_resolution)
    ph.print_info(
        "Isotropic reconstruction space with %g mm resolution created" %
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
            "Isotropic reconstruction space centered around joint stack masks")

        # Crop to space defined by mask (plus extra margin)
        HR_volume = HR_volume.get_cropped_stack_based_on_mask(
            boundary_i=args.extra_frame_target,
            boundary_j=args.extra_frame_target,
            boundary_k=args.extra_frame_target,
            unit="mm",
        )

        # Scattered Data Approximation to get first estimate of HR volume
        ph.print_title("First Estimate of HR Volume")
        SDA = sda.ScatteredDataApproximation(
            stacks, HR_volume, sigma=args.sigma)
        SDA.run()
        HR_volume = SDA.get_reconstruction()

    time_reconstruction = ph.stop_timing(time_tmp)

    if args.verbose:
        tmp = list(stacks)
        tmp.insert(0, HR_volume)
        sitkh.show_stacks(tmp, segmentation=HR_volume)

    # ----------------Two-step Slice-to-Volume Registration SRR----------------
    SRR = tk.TikhonovSolver(
        stacks=stacks,
        reconstruction=HR_volume,
        reg_type="TK1",
        minimizer="lsmr",
        alpha=args.alpha_first,
        iter_max=args.iter_max_first,
        verbose=True,
    )

    if args.two_step_cycles > 0:

        registration = regsitk.SimpleItkRegistration(
            moving=HR_volume,
            use_fixed_mask=True,
            use_moving_mask=use_reference_mask,
            use_verbose=args.verbose,
            interpolator="Linear",
            metric="Correlation",
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
            )
        two_step_s2v_reg_recon.run()
        HR_volume_iterations = \
            two_step_s2v_reg_recon.get_iterative_reconstructions()
        time_registration += \
            two_step_s2v_reg_recon.get_computational_time_registration()
        time_reconstruction += \
            two_step_s2v_reg_recon.get_computational_time_reconstruction()
    else:
        HR_volume_iterations = []

    # Write motion-correction results
    if args.write_motion_correction:
        for stack in stacks:
            stack.write(
                os.path.join(args.dir_output,
                             args.subfolder_motion_correction),
                write_mask=True,
                write_slices=True,
                write_transforms=True,
                suffix_mask=args.suffix_mask,
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
    HR_volume_final.write(args.dir_output)

    HR_volume_iterations.insert(0, HR_volume_final)
    for stack in stacks:
        HR_volume_iterations.append(stack)

    if args.verbose and not args.provide_comparison:
        sitkh.show_stacks(HR_volume_iterations, segmentation=HR_volume)
    # HR_volume_final.show()

    # Show SRR together with linearly resampled input data.
    # Additionally, a script is generated to open files
    if args.provide_comparison:
        sitkh.show_stacks(HR_volume_iterations,
                          segmentation=HR_volume,
                          show_comparison_file=args.provide_comparison,
                          dir_output=os.path.join(
                              args.dir_output,
                              args.subfolder_comparison),
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
