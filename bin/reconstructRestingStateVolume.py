#!/usr/bin/python

##
# \file reconstructVolume.py
# \brief      Script to reconstruct an isotropic, high-resolution volume from
#             multiple stacks of low-resolution 2D slices including
#             motion-correction.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       March 2016
#

# Import libraries
import SimpleITK as sitk
import argparse
import numpy as np
import sys
import os

import pythonhelper.PythonHelper as ph
import pythonhelper.SimpleITKHelper as sitkh

import volumetricreconstruction.base.DataReader as dr
import volumetricreconstruction.base.Stack as st
import volumetricreconstruction.preprocessing.DataPreprocessing as dp
import volumetricreconstruction.preprocessing.N4BiasFieldCorrection as n4bfc
import volumetricreconstruction.registration.RegistrationSimpleITK as regsitk
import volumetricreconstruction.registration.RegistrationCppITK as regitk
import volumetricreconstruction.registration.FLIRT as regflirt
import volumetricreconstruction.registration.NiftyReg as regniftyreg
import volumetricreconstruction.registration.SegmentationPropagation as segprop
import volumetricreconstruction.reconstruction.ScatteredDataApproximation as \
    sda
import volumetricreconstruction.reconstruction.solver.TikhonovSolver as tk
import volumetricreconstruction.reconstruction.solver.ADMMSolver as admm
import volumetricreconstruction.utilities.Exceptions as Exceptions
import volumetricreconstruction.utilities.VolumetricReconstructionPipeline as \
    pipeline
from volumetricreconstruction.utilities.InputArparser import InputArgparser
import volumetricreconstruction.base.DataWriter as dw

if __name__ == '__main__':

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
        prog="python " + os.path.basename(__file__),
    )
    input_parser.add_filename()
    input_parser.add_filename_mask()
    input_parser.add_dir_output(default="results/")
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_prefix_output(default="SRR_")
    input_parser.add_target_stack_index(default=0)
    input_parser.add_sigma(default=0.8)
    input_parser.add_reg_type(default="TK1")
    input_parser.add_alpha_first(default=0.08)
    input_parser.add_alpha(default=0.05)
    input_parser.add_iter_max_first(default=5)
    input_parser.add_iter_max(default=10)
    input_parser.add_minimizer(default="lsmr")
    input_parser.add_data_loss(default="linear")
    input_parser.add_dilation_radius(default=3)
    input_parser.add_extra_frame_target(default=5)
    input_parser.add_bias_field_correction(default=0)
    input_parser.add_intensity_correction(default=0)
    input_parser.add_isotropic_resolution(default=1)
    input_parser.add_log_script_execution(default=1)
    input_parser.add_write_motion_correction(default=1)
    input_parser.add_provide_comparison(default=1)
    input_parser.add_verbose(default=1)
    input_parser.add_two_step_cycles(default=1)
    input_parser.add_rho(default=0.5)
    input_parser.add_admm_iterations(default=10)
    input_parser.add_stack_recon_range(default=15)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # Write script execution call
    if args.log_script_execution:
        performed_script_execution = ph.get_performed_script_execution(
            os.path.basename(__file__), args)
        ph.write_performed_script_execution_to_executable_file(
            performed_script_execution,
            os.path.join(args.dir_output,
                         "log_%s_script_execution.sh" % (
                             os.path.basename(__file__).split(".")[0])))

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")
    data_reader = dr.MultiComponentImageReader(
        args.filename, args.filename_mask)
    data_reader.read_data()
    stacks = data_reader.get_stacks()

    # ------------------------------DELETE LATER------------------------------
    # stacks = stacks[0:3]
    # ------------------------------DELETE LATER------------------------------

    # ---------------------------Data Preprocessing---------------------------
    ph.print_title("Data Preprocessing")

    segmentation_propagator = segprop.SegmentationPropagation(
        # registration_method=regniftyreg.RegAladin(use_verbose=args.verbose),
        # registration_method=regsitk.RegistrationSimpleITK(use_verbose=args.verbose),
        # registration_method=regitk.RegistrationCppITK(use_verbose=args.verbose),
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
    data_preprocessing.run_preprocessing()
    time_data_preprocessing = data_preprocessing.get_computational_time()

    # Get preprocessed stacks
    stacks = data_preprocessing.get_preprocessed_stacks()

    # if args.verbose:
    #     sitkh.show_stacks(stacks, segmentation=stacks[0])

    # ------------------------Volume-to-Volume Registration--------------------
    if args.two_step_cycles > 0:

        # registration = regniftyreg.RegAladin(
        registration = regflirt.FLIRT(
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

    if args.reg_type in ["TK0", "TK1"]:
        reconstruction_method = tk.TikhonovSolver(
            stacks=None,
            reconstruction=st.Stack.from_stack(
                stacks[args.target_stack_index]),
            reg_type=args.reg_type,
        )
    elif args.reg_type == "TV":
        reconstruction_method = admm.ADMMSolver(
            stacks=stacks,
            reconstruction=st.Stack.from_stack(
                stacks[args.target_stack_index]),
            rho=args.rho,
            iterations=args.ADMM_iterations,
        )
    reconstruction_method.set_alpha(args.alpha)
    reconstruction_method.set_iter_max(args.iter_max)
    reconstruction_method.set_minimizer(args.minimizer)
    reconstruction_method.set_data_loss(args.data_loss)
    reconstruction_method.set_verbose(args.verbose)

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

    # ---------------------------Create first volume---------------------------
    time_tmp = ph.start_timing()
    # Isotropic resampling to define HR target space
    ph.print_title("Isotropic Resampling")
    HR_volume = stacks[args.target_stack_index].\
        get_isotropically_resampled_stack(
        spacing_new_scalar=args.isotropic_resolution,
        extra_frame=args.extra_frame_target)
    HR_volume.set_filename(
        stacks[args.target_stack_index].get_filename() +
        "_upsampled")

    # Scattered Data Approximation to get first estimate of HR volume
    ph.print_title("Scattered Data Approximation")
    SDA = sda.ScatteredDataApproximation(
        stacks[args.target_stack_index: args.stack_recon_range],
        HR_volume, sigma=args.sigma)
    SDA.run_reconstruction()
    SDA.generate_mask_from_stack_mask_unions(
        mask_dilation_radius=2, mask_dilation_kernel="Ball")
    HR_volume = SDA.get_reconstruction()
    HR_volume.set_filename(SDA.get_setting_specific_filename())

    time_reconstruction += ph.stop_timing(time_tmp)

    # ----------------Two-step Slice-to-Volume Registration SRR----------------
    if args.two_step_cycles > 0:

        # Two-step registration reconstruction
        registration = regsitk.RegistrationSimpleITK(
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
            initializer_type=None,
            optimizer="ConjugateGradientLineSearch",
            optimizer_params={
                "learningRate": 1,
                "numberOfIterations": 100,
            },
            # optimizer="RegularStepGradientDescent",
            # optimizer_params={
            #     "minStep": 1e-6,
            #     "numberOfIterations": 200,
            #     "gradientMagnitudeTolerance": 1e-6,
            #     "learningRate": 1,
            # },
        )

        alphas = np.linspace(args.alpha_first, args.alpha,
                             args.two_step_cycles + 1)

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
                alphas=alphas[0:args.two_step_cycles],
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
    reconstruction_method.run_reconstruction()
    time_reconstruction += reconstruction_method.get_computational_time()

    HR_volume = reconstruction_method.get_reconstruction()
    HR_volume.set_filename(
        reconstruction_method.get_setting_specific_filename())
    HR_volume_iterations.insert(0, HR_volume)

    for stack in HR_volume_iterations:
        stack.write(args.dir_output)

    # -----------------------Build multi-component image-----------------------
    reconstruction_method.set_reconstruction(st.Stack.from_stack(
        stacks[args.target_stack_index]))
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
