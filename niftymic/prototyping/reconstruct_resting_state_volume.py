##
# \file reconstruct_resting_state_volume.py
# \brief      { item_description }
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       March 2017
#

# Import libraries
import numpy as np
import os

import niftymic.base.data_reader as dr
import niftymic.base.data_writer as dw
import niftymic.base.stack as st
import niftymic.reconstruction.primal_dual_solver as pd
import niftymic.reconstruction.scattered_data_approximation as \
    sda
import niftymic.reconstruction.tikhonov_solver as tk
import niftymic.registration.niftyreg as regniftyreg
import niftymic.utilities.segmentation_propagation as segprop
import niftymic.registration.simple_itk_registration as regsitk
import niftymic.utilities.data_preprocessing as dp
import niftymic.utilities.volumetric_reconstruction_pipeline as \
    pipeline
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
        "slices WITH motion correction. The resolution of the computed "
        "Super-Resolution Reconstruction (SRR) is given by the in-plane "
        "spacing of the selected target stack. A region of interest can be "
        "specified by providing a mask for the selected target stack. Only "
        "this region will then be reconstructed by the SRR algorithm which "
        "can substantially reduce the computational time.",
    )
    input_parser.add_filename()
    input_parser.add_filename_mask()
    input_parser.add_dir_output(default="results/")
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_prefix_output(default="SRR_")
    input_parser.add_target_stack_index(default=0)
    input_parser.add_sigma(default=0.8)
    input_parser.add_alpha_first(default=0.05)
    input_parser.add_alpha(default=0.03)
    input_parser.add_iter_max_first(default=5)
    input_parser.add_iter_max(default=10)
    input_parser.add_reconstruction_type(default="TK1L2")
    input_parser.add_minimizer(default="lsmr")
    input_parser.add_data_loss(default="linear")
    input_parser.add_dilation_radius(default=3)
    input_parser.add_extra_frame_target(default=5)
    input_parser.add_bias_field_correction(default=0)
    input_parser.add_intensity_correction(default=0)
    input_parser.add_isotropic_resolution(default=1)
    input_parser.add_log_config(default=1)
    input_parser.add_write_motion_correction(default=1)
    input_parser.add_provide_comparison(default=1)
    input_parser.add_verbose(default=1)
    input_parser.add_two_step_cycles(default=1)
    input_parser.add_rho(default=0.5)
    input_parser.add_iterations(default=10)
    input_parser.add_stack_recon_range(default=15)
    input_parser.add_option(
        option_string="--reconstruction-spacing",
        type=float,
        nargs="+",
        help="Specify spacing of reconstruction space in case a change is desired",
        default=None)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

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
    reconstruction_space = st.Stack.from_stack(stacks[args.target_stack_index])
    if args.reconstruction_spacing is not None:
        reconstruction_space = reconstruction_space.get_resampled_stack(
            spacing=args.reconstruction_spacing)

    # ------------------------------DELETE LATER------------------------------
    # stacks = stacks[0:3]
    # ------------------------------DELETE LATER------------------------------

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

    # if args.verbose:
    #     sitkh.show_stacks(stacks, segmentation=stacks[0])

    # ------------------------Volume-to-Volume Registration--------------------
    if args.two_step_cycles > 0:

        registration = regniftyreg.RegAladin(
            # registration = regflirt.FLIRT(
            fixed=stacks[args.target_stack_index],
            registration_type="Rigid",
            use_fixed_mask=True,
            use_moving_mask=True,
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

    reconstruction_method = tk.TikhonovSolver(
        stacks=stacks,
        reconstruction=reconstruction_space,
        reg_type="TK1",
        alpha=args.alpha,
        iter_max=args.iter_max,
        minimizer=args.minimizer,
        data_loss=args.data_loss,
        verbose=args.verbose,
    )

    multi_component_reconstruction = pipeline.MultiComponentReconstruction(
        stacks=stacks,
        reconstruction_method=reconstruction_method,
        suffix="_recon_v2v")
    multi_component_reconstruction.run()
    time_reconstruction = \
        multi_component_reconstruction.get_computational_time()
    stacks_recon_v2v = multi_component_reconstruction.get_reconstructions()

    # Write result
    filename = os.path.basename(args.filename).split(".")[0]
    filename = os.path.join(args.dir_output, filename + "_recon_v2v.nii.gz")
    data_writer = dw.MultiComponentImageWriter(stacks_recon_v2v, filename)
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
        stacks[args.target_stack_index: args.stack_recon_range],
        HR_volume, sigma=args.sigma)
    SDA.run()
    SDA.generate_mask_from_stack_mask_unions(
        mask_dilation_radius=2, mask_dilation_kernel="Ball")
    HR_volume = SDA.get_reconstruction()
    HR_volume.set_filename(SDA.get_setting_specific_filename())

    time_reconstruction += ph.stop_timing(time_tmp)

    # ----------------Two-step Slice-to-Volume Registration SRR-------------
    if args.two_step_cycles > 0:

        # Two-step registration reconstruction
        registration = regsitk.SimpleItkRegistration(
            moving=HR_volume,
            use_fixed_mask=True,
            use_moving_mask=True,
            use_verbose=args.verbose,
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

        reconstruction_method.set_stacks(
            stacks[args.target_stack_index: args.stack_recon_range])
        reconstruction_method.set_reconstruction(HR_volume)
        reconstruction_method.set_iter_max(args.iter_max_first)

        two_step_s2v_reg_recon = \
            pipeline.TwoStepSliceToVolumeRegistrationReconstruction(
                stacks=stacks,
                reference=HR_volume,
                registration_method=registration,
                reconstruction_method=reconstruction_method,
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

        if args.verbose:
            sitkh.show_stacks(HR_volume_iterations)

        # # Write to output
        # HR_volume_tmp.write(args.dir_output)

    ph.print_title("Final Super-Resolution Reconstruction")
    reconstruction_method.set_alpha(args.alpha)
    reconstruction_method.set_iter_max(args.iter_max)
    reconstruction_method.run()
    time_reconstruction += reconstruction_method.get_computational_time()

    HR_volume = reconstruction_method.get_reconstruction()
    HR_volume.set_filename(
        reconstruction_method.get_setting_specific_filename())
    HR_volume_iterations.insert(0, HR_volume)

    for stack in HR_volume_iterations:
        stack.write(args.dir_output)

    # -----------------------Build multi-component image-----------------------
    if args.reconstruction_type in ["TVL2", "HuberL2"]:
        reconstruction_method = pd.PrimalDualSolver(
            stacks=stacks,
            reconstruction=reconstruction_space,
            reg_type="TV" if args.reconstruction_type == "TVL2" else "huber",
            iterations=args.iterations,
        )
    else:
        reconstruction_method = tk.TikhonovSolver(
            stacks=stacks,
            reconstruction=reconstruction_space,
            reg_type="TK1" if args.reconstruction_type == "TK1L2" else "TK0",
        )
    reconstruction_method.set_alpha(args.alpha)
    reconstruction_method.set_iter_max(args.iter_max)
    reconstruction_method.set_verbose(args.verbose)
    # multi_component_reconstruction = pipeline.MultiComponentReconstruction(
    #     stacks=stacks,
    #     reconstruction_method=reconstruction_method,
    #     suffix="_recon_v2v")
    multi_component_reconstruction.set_reconstruction_method(
        reconstruction_method)
    multi_component_reconstruction.set_suffix("_recon_s2v")
    multi_component_reconstruction.run()

    time_reconstruction += \
        multi_component_reconstruction.get_computational_time()

    stacks_recon_s2v = multi_component_reconstruction.get_reconstructions()

    # Write result
    filename = os.path.basename(args.filename).split(".")[0]
    filename = os.path.join(args.dir_output, filename + "_recon_s2v.nii.gz")
    data_writer = dw.MultiComponentImageWriter(stacks_recon_s2v, filename)
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
