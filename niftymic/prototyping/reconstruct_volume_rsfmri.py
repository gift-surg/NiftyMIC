##
# \file reconstruct_volume_rsfmri.py
# \brief      Script to run the slice-based motion-correction algorithm for
#             resting-state fMRI
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
        description="Volumetric MRI reconstruction framework to reconstruct "
        "resting-state fMRI using slice-based motion-correction algorithms.",
    )
    input_parser.add_filename(required=True)
    input_parser.add_filename_mask()
    input_parser.add_dir_output(required=True)

    input_parser.add_option(
        option_string="--alpha-rsfmri",
        type=float,
        help="Regularization parameter used for rsfmri reconstruction",
        default=0.05,
    )
    input_parser.add_alpha(default=0.03)
    input_parser.add_alpha_first(default=0.05)
    input_parser.add_data_loss(default="linear")
    input_parser.add_dilation_radius(default=3)
    input_parser.add_extra_frame_target(default=0)
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
    input_parser.add_option(
        option_string="--reconstruction-spacing",
        type=float,
        nargs="+",
        help="Specify spacing of reconstruction space in case a change is desired",
        default=None)
    input_parser.add_reconstruction_type(default="TK1L2")
    input_parser.add_rho(default=0.5)
    input_parser.add_sigma(default=0.8)
    input_parser.add_stack_recon_range(default=15)
    input_parser.add_target_stack_index(default=0)
    input_parser.add_two_step_cycles(default=3)
    input_parser.add_use_masks_srr(default=1)
    input_parser.add_verbose(default=0)
    input_parser.add_v2v_method(default="RegAladin")
    input_parser.add_outlier_rejection(default=1)
    input_parser.add_threshold_first(default=0.5)
    input_parser.add_threshold(default=0.8)
    input_parser.add_write_motion_correction(default=1)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    if args.v2v_method not in V2V_METHOD_OPTIONS:
        raise ValueError("v2v-method must be in {%s}" % (
            ", ".join(V2V_METHOD_OPTIONS)))

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

    # Define reconstruction space for rsfmri
    reconstruction_space = st.Stack.from_sitk_image(
        stacks[args.target_stack_index].sitk * 0,
        slice_thickness=stacks[args.target_stack_index].sitk.GetSpacing()[-1],
        filename=stacks[args.target_stack_index].get_filename(),
    )
    if args.reconstruction_spacing is not None:
        reconstruction_space = reconstruction_space.get_resampled_stack(
            spacing=args.reconstruction_spacing)

    # ------------------------------DELETE LATER------------------------------
    if args.prototyping:
        stacks = stacks[0:2]
    # ------------------------------DELETE LATER------------------------------

    # specify stack index range used for intermediate volumetric HR recons
    i_min = args.target_stack_index
    i_max = np.min([args.target_stack_index + args.stack_recon_range,
                    len(stacks)])

    # ---------------------------Data Preprocessing---------------------------
    ph.print_title("Data Preprocessing")

    segmentation_propagator = segprop.SegmentationPropagation(
        # registration_method=regniftyreg.RegAladin(use_verbose=args.verbose),
        # registration_method=regsitk.SimpleItkRegistration(use_verbose=args.verbose),
        # registration_method=regitk.CppItkRegistration(use_verbose=args.verbose),
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

    # if args.verbose:
    #     sitkh.show_stacks(stacks, segmentation=stacks[0])

    # ------------------------Volume-to-Volume Registration--------------------
    if args.two_step_cycles > 0:

        if args.v2v_method == "FLIRT":
            registration = regflirt.FLIRT(
                fixed=stacks[args.target_stack_index],
                registration_type="Rigid",
                use_fixed_mask=True,
                use_moving_mask=True,
                use_verbose=False,
            )
        else:
            registration = regniftyreg.RegAladin(
                fixed=stacks[args.target_stack_index],
                registration_type="Rigid",
                use_fixed_mask=True,
                use_moving_mask=True,
                # options="-ln 2",
                use_verbose=False,
            )

        v2vreg = pipeline.VolumeToVolumeRegistration(
            stacks=stacks,
            reference=stacks[args.target_stack_index],
            registration_method=registration,
            verbose=args.verbose,
        )
        v2vreg.run()
        stacks = v2vreg.get_stacks()
        time_registration = v2vreg.get_computational_time()

    else:
        time_registration = ph.get_zero_time()

    # ----Define solver for rsfMRI reconstructions of individual timepoints----
    if args.reconstruction_type in ["TVL2", "HuberL2"]:
        reconstruction_method_rsfmri = pd.PrimalDualSolver(
            stacks=stacks,
            reconstruction=reconstruction_space,
            reg_type="TV" if args.reconstruction_type == "TVL2" else "huber",
            iterations=args.iterations,
        )
    else:
        reconstruction_method_rsfmri = tk.TikhonovSolver(
            stacks=stacks,
            reconstruction=reconstruction_space,
            reg_type="TK1" if args.reconstruction_type == "TK1L2" else "TK0",
        )
    reconstruction_method_rsfmri.set_alpha(args.alpha_rsfmri)
    reconstruction_method_rsfmri.set_iter_max(args.iter_max)
    reconstruction_method_rsfmri.set_verbose(args.verbose)
    reconstruction_method_rsfmri.set_minimizer(args.minimizer)
    reconstruction_method_rsfmri.set_data_loss(args.data_loss)
    reconstruction_method_rsfmri.set_use_masks(args.use_masks_srr)

    # ------Update individual timepoints based on updated slice positions------
    multi_component_reconstruction = pipeline.MultiComponentReconstruction(
        stacks=stacks,
        reconstruction_method=reconstruction_method_rsfmri,
        suffix="_recon_v2v")
    multi_component_reconstruction.run()
    time_reconstruction = \
        multi_component_reconstruction.get_computational_time()
    stacks_recon_v2v = multi_component_reconstruction.get_reconstructions()
    description = multi_component_reconstruction.get_reconstruction_method().\
        get_setting_specific_filename()

    # Write result
    filename = os.path.basename(args.filename).split(".")[0]
    path_to_v2v = os.path.join(args.dir_output, filename + "_recon_v2v.nii.gz")
    data_writer = dw.MultiComponentImageWriter(
        stacks_recon_v2v, path_to_v2v, description=description)
    data_writer.write_data()

    # ---------------------------Create first volume------------------------
    time_tmp = ph.start_timing()
    # Isotropic resampling to define HR target space
    ph.print_title("Isotropic Resampling")
    HR_volume = stacks[args.target_stack_index].\
        get_isotropically_resampled_stack(
        resolution=args.isotropic_resolution,
        extra_frame=args.extra_frame_target)
    HR_volume.set_filename(
        stacks[args.target_stack_index].get_filename() +
        "_upsampled")

    # Scattered Data Approximation to get first estimate of HR volume
    ph.print_title("Scattered Data Approximation")
    SDA = sda.ScatteredDataApproximation(
        stacks=stacks[i_min: i_max],
        HR_volume=HR_volume,
        sigma=args.sigma,
    )
    SDA.run()
    HR_volume = SDA.get_reconstruction()

    joint_image_mask_builder = imb.JointImageMaskBuilder(
        stacks=stacks[i_min: i_max],
        target=HR_volume,
        dilation_radius=1,
    )
    joint_image_mask_builder.run()
    HR_volume = joint_image_mask_builder.get_stack()
    HR_volume.set_filename(SDA.get_setting_specific_filename())

    # Crop to space defined by mask (plus extra margin)
    HR_volume = HR_volume.get_cropped_stack_based_on_mask(
        boundary_i=args.extra_frame_target,
        boundary_j=args.extra_frame_target,
        boundary_k=args.extra_frame_target,
        unit="mm",
    )

    time_reconstruction += ph.stop_timing(time_tmp)

    # ----------------Two-step Slice-to-Volume Registration SRR-------------
    if args.two_step_cycles > 0:
        stacks_srr = [st.Stack.from_stack(s) for s in stacks[i_min: i_max]]

        # Define the regularization parameters for the individual
        # reconstruction steps in the two-step cycles
        alphas = np.linspace(
            args.alpha_first, args.alpha, args.two_step_cycles)

        # Two-step registration reconstruction
        registration = regsitk.SimpleItkRegistration(
            moving=HR_volume,
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

        # Use "standard" SRR algorithm to create SRR reference
        reconstruction_method_srr = tk.TikhonovSolver(
            stacks=stacks_srr,
            reconstruction=HR_volume,
            reg_type="TK1",
            iter_max=np.min([args.iter_max_first, args.iter_max]),
            use_masks=False,
        )

        # Define outlier rejection threshold after each S2V-reg step
        thresholds = np.linspace(
            args.threshold_first, args.threshold, args.two_step_cycles)

        two_step_s2v_reg_recon = \
            pipeline.TwoStepSliceToVolumeRegistrationReconstruction(
                stacks=stacks_srr,
                reference=HR_volume,
                registration_method=registration,
                reconstruction_method=reconstruction_method_srr,
                cycles=args.two_step_cycles,
                alphas=alphas[0:args.two_step_cycles - 1],
                verbose=args.verbose,
                outlier_rejection=args.outlier_rejection,
                thresholds=thresholds,
            )
        two_step_s2v_reg_recon.run()
        HR_volume_iterations = \
            two_step_s2v_reg_recon.get_iterative_reconstructions()
        time_registration += \
            two_step_s2v_reg_recon.get_computational_time_registration()
        time_reconstruction += \
            two_step_s2v_reg_recon.get_computational_time_reconstruction()

        if args.verbose:
            sitkh.show_stacks(HR_volume_iterations)

        # # Write to output
        # HR_volume_tmp.write(args.dir_output)

    ph.print_title("Final Super-Resolution Reconstruction")
    reconstruction_method_srr.set_alpha(args.alpha)
    reconstruction_method_srr.set_iter_max(args.iter_max)
    reconstruction_method_srr.run()
    time_reconstruction += reconstruction_method_srr.get_computational_time()

    HR_volume = reconstruction_method_srr.get_reconstruction()
    description = reconstruction_method_srr.get_setting_specific_filename()
    HR_volume.set_filename(description)
    dw.DataWriter.write_image(
        image_sitk=HR_volume.sitk,
        path_to_file=os.path.join(args.dir_output, "SRR_reference.nii.gz"),
        description=description,
    )

    # for stack in HR_volume_iterations:
    #     stack.write(args.dir_output)

    # --------------------Final Slice-to-Volume Registrations-----------------
    ph.print_title("Final Slice-to-Volume Registrations")
    s2vreg = pipeline.SliceToVolumeRegistration(
        stacks=stacks,
        reference=HR_volume,
        registration_method=registration,
        verbose=False,
    )
    s2vreg.run()

    # ------------------Write Slice Motion Correction Results------------------
    ph.print_title("Write Slice Motion Correction Results")
    if args.write_motion_correction:
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

    # ------Update individual timepoints based on updated slice positions------
    multi_component_reconstruction.set_suffix("_recon_s2v")
    multi_component_reconstruction.run()

    time_reconstruction += \
        multi_component_reconstruction.get_computational_time()

    stacks_recon_s2v = multi_component_reconstruction.get_reconstructions()
    description = multi_component_reconstruction.get_reconstruction_method().\
        get_setting_specific_filename()

    # -----------------------Write multi-component image-----------------------
    filename = os.path.basename(args.filename).split(".")[0]
    path_to_s2v = os.path.join(args.dir_output, filename + "_recon_s2v.nii.gz")
    data_writer = dw.MultiComponentImageWriter(
        stacks_recon_s2v, path_to_s2v, description=description)
    data_writer.write_data()

    # if args.verbose:
    #     for i in range(0, len(stacks)):
    #         sitkh.show_stacks([stacks_recon_v2v[i], stacks_recon_s2v[i]])
    #         ph.pause()
    #         ph.killall_itksnap()

    elapsed_time_total = ph.stop_timing(time_start)

    # Show SRR together with linearly resampled input data.
    # Additionally, a script is generated to open files
    # if args.provide_comparison:
    #     sitkh.show_stacks(HR_volume_iterations,
    #                       show_comparison_file=args.provide_comparison,
    #                       dir_output=os.path.join(
    #                           args.dir_output, "comparison"),
    #                       )

    if args.verbose:
        ph.show_niftis([
            args.filename,
            path_to_v2v,
            path_to_s2v,
        ])

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
