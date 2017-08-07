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
import volumetricreconstruction.registration.RegistrationITK as regitk
import volumetricreconstruction.registration.FLIRT as regflirt
import volumetricreconstruction.registration.NiftyReg as regniftyreg
import volumetricreconstruction.registration.SegmentationPropagation as segprop
import volumetricreconstruction.reconstruction.ScatteredDataApproximation as sda
import volumetricreconstruction.reconstruction.solver.TikhonovSolver as tk
import volumetricreconstruction.reconstruction.solver.ADMMSolver as admm
import volumetricreconstruction.utilities.Exceptions as Exceptions
from volumetricreconstruction.utilities.InputArparser import InputArgparser


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
    input_parser.add_prefix_output(default="_SRR")
    input_parser.add_target_stack_index(default=0)
    input_parser.add_sigma(default=0.9)
    input_parser.add_alpha(default=0.03)
    input_parser.add_alpha_first(default=0.1)
    input_parser.add_reg_type(default="TK1")
    input_parser.add_iter_max(default=10)
    input_parser.add_iter_max_first(default=5)
    input_parser.add_minimizer(default="lsmr")
    input_parser.add_data_loss(default="linear")
    input_parser.add_dilation_radius(default=3)
    input_parser.add_extra_frame_target(default=10)
    input_parser.add_bias_field_correction(default=0)
    input_parser.add_intensity_correction(default=0)
    input_parser.add_isotropic_resolution(default=1)
    input_parser.add_log_script_execution(default=1)
    input_parser.add_write_motion_correction(default=1)
    input_parser.add_provide_comparison(default=1)
    input_parser.add_verbose(default=1)
    input_parser.add_two_step_cycles(default=3)
    input_parser.add_rho(default=0.5)
    input_parser.add_admm_iterations(default=10)

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

    # Read Data:
    ph.print_title("Read Data")
    data_reader = dr.MultiComponentImageReader(
        args.filename, args.filename_mask)
    data_reader.read_data()
    stacks = data_reader.get_stacks()

    # ------------------------------DELETE LATER------------------------------
    # stacks = stacks[0:20]
    # ------------------------------DELETE LATER------------------------------

    # Data Preprocessing from data on HDD
    ph.print_title("Data Preprocessing")

    segmentation_propagator = segprop.SegmentationPropagation(
        # registration_method=regniftyreg.NiftyReg(use_verbose=args.verbose),
        # registration_method=regsitk.RegistrationSimpleITK(use_verbose=args.verbose),
        # registration_method=regitk.RegistrationITK(use_verbose=args.verbose),
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

    if args.two_step_cycles > 0:

        time_registration = ph.start_timing()

        # Global rigid registration to target stack
        ph.print_title("Global Rigid Registration")

        # registration = regsitk.RegistrationSimpleITK(
        #     initializer_type="GEOMETRY", metric="MattesMutualInformation",
        registration = regniftyreg.NiftyReg(
            # registration = regflirt.FLIRT(
            fixed=stacks[0],
            registration_type="Rigid",
            use_fixed_mask=True,
            use_moving_mask=True,
            use_verbose=False,
        )

        for i in range(1, len(stacks)):
            registration.set_moving(stacks[i])
            registration.run_registration()
            transform_sitk = registration.get_registration_transform_sitk()
            transform_sitk = eval(
                "sitk." + transform_sitk.GetName() +
                "(transform_sitk.GetInverse())")
            stacks[i].update_motion_correction(transform_sitk)

        time_registration = ph.stop_timing(time_registration)
        # sitkh.show_stacks(stacks, segmentation=stacks[0])

        # if args.verbose:
        #     for i in range(0, len(stacks)):
        #         stacks[i].write(
        #             directory=os.path.join(args.dir_output,
        #                                    "02_rigidly_aligned_data"),
        #             write_mask=True)

    else:
        time_registration = 0

    time_reconstruction = ph.start_timing()

    # Isotropic resampling to define HR target space
    ph.print_title("Isotropic Resampling")
    HR_volume = stacks[args.target_stack_index].\
        get_isotropically_resampled_stack(
        spacing_new_scalar=args.isotropic_resolution,
        extra_frame=args.extra_frame_target)
    HR_volume.set_filename(
        "Iter0_" + stacks[args.target_stack_index].get_filename() +
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
    HR_volume.set_filename("Iter0_" + SDA.get_setting_specific_filename())

    time_reconstruction = ph.stop_timing(time_reconstruction)

    # List to store SRR iterations
    HR_volume_iterations = []

    # Add initial volume and rigidly aligned, original data for
    # visualization
    HR_volume_iterations.append(st.Stack.from_stack(HR_volume))
    # for i in range(args.target_stack_index,
    #                args.target_stack_index + args.stack_recon_range):
    #     HR_volume_iterations.append(stacks[i])
    HR_volume.write(args.dir_output)

    if args.verbose:
        sitkh.show_stacks(HR_volume_iterations)

    if args.reg_type in ["TK0", "TK1"]:
        SRR = tk.TikhonovSolver(
            stacks=stacks[args.target_stack_index: args.stack_recon_range],
            reconstruction=HR_volume,
            alpha=args.alpha_first,
            iter_max=args.iter_max_first,
            reg_type=args.reg_type,
            minimizer=args.minimizer,
            data_loss=args.data_loss,
            verbose=args.verbose,
        )
    elif args.reg_type == "TV":
        SRR = admm.ADMMSolver(
            stacks=stacks,
            reconstruction=HR_volume,
            alpha=args.alpha_first,
            minimizer=args.minimizer,
            iter_max=args.iter_max_first,
            rho=args.rho,
            iterations=args.ADMM_iterations,
            verbose=args.verbose,
        )

    if args.two_step_cycles > 0:

        alpha_delta = (args.alpha - args.alpha_first) / \
            float(args.two_step_cycles)

        # Two-step Slice-to-Volume Registration Reconstruction
        ph.print_title("Two-step Slice-to-Volume Registration Reconstruction")

        # Two-step registration reconstruction
        registration = regsitk.RegistrationSimpleITK(
            moving=HR_volume,
            use_fixed_mask=True,
            use_moving_mask=False,
            use_verbose=args.verbose,
            interpolator="Linear",
            # metric="Correlation",
            metric="MattesMutualInformation",  # Might cause error messages
            # like "Too many samples map outside moving image buffer."
            # use_multiresolution_framework=True,
            shrink_factors=[2, 1],
            smoothing_sigmas=[1, 0],
            initializer_type=None,
            # optimizer="RegularStepGradientDescent",
            # optimizer_params="{'learningRate': 1, 'minStep': 1e-6,\
            # 'numberOfIterations': 600, 'gradientMagnitudeTolerance': 1e-6}",
            optimizer="ConjugateGradientLineSearch",
            optimizer_params="{'learningRate': 1, 'numberOfIterations': 100}",
        )
        # registration = regflirt.FLIRT(
        #     moving=HR_volume,
        #     registration_type="Rigid",
        #     use_fixed_mask=True,
        #     use_moving_mask=False,
        #     use_verbose=args.verbose,
        #     options="-cost normcorr -searchcost normcorr -nosearch",
        # )

        for i_cycle in range(0, args.two_step_cycles):
            time_elapsed_tmp = ph.start_timing()
            for i_stack in range(0, len(stacks)):
                stack = stacks[i_stack]

                # Slice-to-volume registration
                for i_slice in range(0, stack.get_number_of_slices()):
                    txt = "Cycle %d/%d -- Stack %d/%d -- Slice %2d/%d: " \
                        "Slice-to-Volume Registration" % (
                            i_cycle+1, args.two_step_cycles, i_stack+1,
                            len(stacks), i_slice+1,
                            stack.get_number_of_slices())
                    if args.verbose:
                        ph.print_subtitle(txt)
                    else:
                        ph.print_info(txt)
                    slice = stack.get_slice(i_slice)
                    registration.set_fixed(slice)
                    registration.run_registration()
                    transform_sitk = \
                        registration.get_registration_transform_sitk()
                    slice.update_motion_correction(transform_sitk)
            time_elapsed_tmp = ph.stop_timing(time_elapsed_tmp)
            time_registration = ph.add_times(
                time_registration, time_elapsed_tmp)
            print("\nElapsed time for all Slice-to-Volume registrations: %s"
                  % (time_elapsed_tmp))

            # Super-resolution reconstruction
            time_elapsed_tmp = ph.start_timing()
            ph.print_subtitle("Cycle %d/%d: Super-Resolution Reconstruction"
                              % (i_cycle+1, args.two_step_cycles))

            SRR.set_alpha(args.alpha + i_cycle*alpha_delta)
            SRR.run_reconstruction()

            time_elapsed_tmp = ph.stop_timing(time_elapsed_tmp)
            SRR.print_statistics()
            time_reconstruction = ph.add_times(
                time_reconstruction, time_elapsed_tmp)

            filename = "Iter" + str(i_cycle+1) + "_" + \
                SRR.get_setting_specific_filename()
            HR_volume_tmp = HR_volume.get_stack_multiplied_with_mask(
                filename=filename)
            HR_volume_iterations.insert(0, HR_volume_tmp)
            if args.verbose:
                sitkh.show_stacks(HR_volume_iterations)

            # Write to output
            HR_volume_tmp.write(args.dir_output)

    ph.print_subtitle("Final Super-Resolution Reconstruction")
    SRR.set_alpha(args.alpha)
    SRR.set_iter_max(args.iter_max)
    stacks_recon = [None] * len(stacks)
    for i in range(0, len(stacks)):
        SRR.set_stacks([st.Stack.from_stack(stacks[i])])
        SRR.set_reconstruction(st.Stack.from_stack(stacks[i]))
        time_elapsed_tmp = ph.start_timing()
        SRR.run_reconstruction()
        time_elapsed_tmp = ph.stop_timing(time_elapsed_tmp)
        time_reconstruction = ph.add_times(time_reconstruction,
                                           time_elapsed_tmp)
        SRR.print_statistics()
        stacks_recon[i] = SRR.get_reconstruction()
        stacks_recon[i].set_filename(stacks[i].get_filename() + "_SRR")

    # if args.verbose:
    #     for i in range(0, len(stacks)):
    #         sitkh.show_stacks([stacks[i], stacks_recon[i]])
    #         ph.pause()
    #         ph.killall_itksnap()

    vector_image_sitk = sitkh.get_sitk_vector_image_from_components(
        image_components_sitk=[v.sitk for v in stacks_recon])

    filename = os.path.basename(args.filename).split(".")[0]
    filename = os.path.join(args.dir_output, filename + "_SRR.nii.gz")
    sitkh.write_sitk_vector_image(vector_image_sitk, filename)

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
