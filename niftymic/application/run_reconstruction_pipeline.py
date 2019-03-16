##
# \file run_reconstruction_pipeline.py
# \brief      Script to execute entire reconstruction pipeline
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       October 2017
#

import os
import re
import numpy as np

import pysitk.python_helper as ph

import niftymic.validation.simulate_stacks_from_reconstruction as \
    simulate_stacks_from_reconstruction
import niftymic.validation.evaluate_simulated_stack_similarity as \
    evaluate_simulated_stack_similarity
import niftymic.validation.show_evaluated_simulated_stack_similarity as \
    show_evaluated_simulated_stack_similarity
import niftymic.application.show_slice_coverage as show_slice_coverage
import niftymic.validation.export_side_by_side_simulated_vs_original_slice_comparison as \
    export_side_by_side_simulated_vs_original_slice_comparison
from niftymic.utilities.input_arparser import InputArgparser
import niftymic.utilities.template_stack_estimator as tse

from niftymic.definitions import DIR_TEMPLATES


def main():

    time_start = ph.start_timing()

    np.set_printoptions(precision=3)

    input_parser = InputArgparser(
        description="Run reconstruction pipeline including "
        "(i) bias field correction, "
        "(ii) volumetric reconstruction in subject space, "
        "(iii) volumetric reconstruction in template space, "
        "and (iv) some diagnostics to assess the obtained reconstruction.",
    )
    input_parser.add_filenames(required=True)
    input_parser.add_filenames_masks(required=True)
    input_parser.add_target_stack(required=False)
    input_parser.add_suffix_mask(default="''")
    input_parser.add_dir_output(required=True)
    input_parser.add_alpha(default=0.01)
    input_parser.add_verbose(default=0)
    input_parser.add_gestational_age(required=False)
    input_parser.add_prefix_output(default="")
    input_parser.add_search_angle(default=180)
    input_parser.add_multiresolution(default=0)
    input_parser.add_log_config(default=1)
    input_parser.add_isotropic_resolution()
    input_parser.add_reference()
    input_parser.add_reference_mask()
    input_parser.add_bias_field_correction(default=1)
    input_parser.add_intensity_correction(default=1)
    input_parser.add_iter_max(default=10)
    input_parser.add_two_step_cycles(default=3)
    input_parser.add_slice_thicknesses(default=None)
    input_parser.add_option(
        option_string="--run-bias-field-correction",
        type=int,
        help="Turn on/off bias field correction. "
        "If off, it is assumed that this step was already performed "
        "if --bias-field-correction is active.",
        default=1)
    input_parser.add_option(
        option_string="--run-recon-subject-space",
        type=int,
        help="Turn on/off reconstruction in subject space. "
        "If off, it is assumed that this step was already performed.",
        default=1)
    input_parser.add_option(
        option_string="--run-recon-template-space",
        type=int,
        help="Turn on/off reconstruction in template space. "
        "If off, it is assumed that this step was already performed.",
        default=1)
    input_parser.add_option(
        option_string="--run-diagnostics",
        type=int,
        help="Turn on/off diagnostics of the obtained volumetric "
        "reconstruction. ",
        default=0)
    input_parser.add_option(
        option_string="--initial-transform",
        type=str,
        help="Set initial transform to be used for register_image.",
        default=None)
    input_parser.add_outlier_rejection(default=1)
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

    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))

    filename_srr = "srr"
    dir_output_preprocessing = os.path.join(
        args.dir_output, "preprocessing_n4itk")
    dir_output_recon_subject_space = os.path.join(
        args.dir_output, "recon_subject_space")
    dir_output_recon_template_space = os.path.join(
        args.dir_output, "recon_template_space")
    dir_output_diagnostics = os.path.join(
        args.dir_output, "diagnostics")

    srr_subject = os.path.join(
        dir_output_recon_subject_space, "%s_subject.nii.gz" % filename_srr)
    srr_subject_mask = ph.append_to_filename(srr_subject, "_mask")
    srr_template = os.path.join(
        dir_output_recon_template_space, "%s_template.nii.gz" % filename_srr)
    srr_template_mask = ph.append_to_filename(srr_template, "_mask")
    trafo_template = os.path.join(
        dir_output_recon_template_space, "registration_transform_sitk.txt")
    srr_slice_coverage = os.path.join(
        dir_output_diagnostics, "%s_template_slicecoverage.nii.gz" % filename_srr)

    if args.target_stack is None:
        target_stack = args.filenames[0]
    else:
        target_stack = args.target_stack

    if args.bias_field_correction and args.run_bias_field_correction:
        for i, f in enumerate(args.filenames):
            output = os.path.join(
                dir_output_preprocessing, os.path.basename(f))
            cmd_args = []
            cmd_args.append("--filename '%s'" % f)
            cmd_args.append("--filename-mask '%s'" % args.filenames_masks[i])
            cmd_args.append("--output '%s'" % output)
            # cmd_args.append("--verbose %d" % args.verbose)
            cmd = "niftymic_correct_bias_field %s" % (" ").join(cmd_args)
            time_start_bias = ph.start_timing()
            exit_code = ph.execute_command(cmd)
            if exit_code != 0:
                raise RuntimeError("Bias field correction failed")
        elapsed_time_bias = ph.stop_timing(time_start_bias)
        filenames = [os.path.join(dir_output_preprocessing, os.path.basename(f))
                     for f in args.filenames]
    elif args.bias_field_correction and not args.run_bias_field_correction:
        elapsed_time_bias = ph.get_zero_time()
        filenames = [os.path.join(dir_output_preprocessing, os.path.basename(f))
                     for f in args.filenames]
    else:
        elapsed_time_bias = ph.get_zero_time()
        filenames = args.filenames

    # Add single quotes around individual filenames to account for whitespaces
    filenames = ["'" + f + "'" for f in filenames]
    filenames_masks = ["'" + f + "'" for f in args.filenames_masks]

    if args.run_recon_subject_space:
        target_stack_index = args.filenames.index(target_stack)

        cmd_args = ["niftymic_reconstruct_volume"]
        cmd_args.append("--filenames %s" % (" ").join(filenames))
        if args.filenames_masks is not None:
            cmd_args.append("--filenames-masks %s" %
                            (" ").join(filenames_masks))
        cmd_args.append("--multiresolution %d" % args.multiresolution)
        cmd_args.append("--target-stack-index %d" % target_stack_index)
        cmd_args.append("--output '%s'" % srr_subject)
        cmd_args.append("--suffix-mask '%s'" % args.suffix_mask)
        cmd_args.append("--intensity-correction %d" %
                        args.intensity_correction)
        cmd_args.append("--alpha %s" % args.alpha)
        cmd_args.append("--iter-max %d" % args.iter_max)
        cmd_args.append("--two-step-cycles %d" % args.two_step_cycles)
        cmd_args.append("--outlier-rejection %d" %
                        args.outlier_rejection)
        if args.slice_thicknesses is not None:
            cmd_args.append("--slice-thicknesses %s" %
                            " ".join(map(str, args.slice_thicknesses)))
        cmd_args.append("--verbose %d" % args.verbose)
        if args.isotropic_resolution is not None:
            cmd_args.append("--isotropic-resolution %f" %
                            args.isotropic_resolution)
        if args.reference is not None:
            cmd_args.append("--reference %s" % args.reference)
        if args.reference_mask is not None:
            cmd_args.append("--reference-mask %s" % args.reference_mask)
        if args.sda:
            cmd_args.append("--sda")
        cmd = (" ").join(cmd_args)
        time_start_volrec = ph.start_timing()
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("Reconstruction in subject space failed")

        # Compute SRR mask in subject space
        # (Approximated using SDA within reconstruct_volume)
        if 0:
            dir_motion_correction = os.path.join(
                dir_output_recon_subject_space, "motion_correction")
            cmd_args = ["niftymic_reconstruct_volume_from_slices"]
            cmd_args.append("--filenames %s" % " ".join(filenames_masks))
            cmd_args.append("--dir-input-mc '%s'" % dir_motion_correction)
            cmd_args.append("--output '%s'" % srr_subject_mask)
            cmd_args.append("--reconstruction-space '%s'" % srr_subject)
            cmd_args.append("--suffix-mask '%s'" % args.suffix_mask)
            cmd_args.append("--mask")
            if args.slice_thicknesses is not None:
                cmd_args.append("--slice-thicknesses %s" %
                                " ".join(map(str, args.slice_thicknesses)))
            if args.sda:
                cmd_args.append("--sda")
                cmd_args.append("--alpha 1")
            else:
                cmd_args.append("--alpha 0.1")
                cmd_args.append("--iter-max 5")
            cmd = (" ").join(cmd_args)
            ph.execute_command(cmd)

        elapsed_time_volrec = ph.stop_timing(time_start_volrec)
    else:
        elapsed_time_volrec = ph.get_zero_time()

    if args.run_recon_template_space:

        if args.gestational_age is None:
            template_stack_estimator = \
                tse.TemplateStackEstimator.from_mask(srr_subject_mask)
            gestational_age = template_stack_estimator.get_estimated_gw()
            ph.print_info("Estimated gestational age: %d" % gestational_age)
        else:
            gestational_age = args.gestational_age

        template = os.path.join(
            DIR_TEMPLATES, "STA%d.nii.gz" % gestational_age)
        template_mask = os.path.join(
            DIR_TEMPLATES, "STA%d_mask.nii.gz" % gestational_age)

        # Register SRR to template space
        cmd_args = ["niftymic_register_image"]
        cmd_args.append("--fixed '%s'" % template)
        cmd_args.append("--moving '%s'" % srr_subject)
        cmd_args.append("--fixed-mask '%s'" % template_mask)
        cmd_args.append("--moving-mask '%s'" % srr_subject_mask)
        cmd_args.append("--dir-input-mc '%s'" % os.path.join(
            dir_output_recon_subject_space, "motion_correction"))
        cmd_args.append("--output '%s'" % trafo_template)
        cmd_args.append("--verbose %s" % args.verbose)
        cmd_args.append("--refine-pca")
        if args.initial_transform is not None:
            cmd_args.append("--initial-transform '%s'" % args.initial_transform)
        cmd = (" ").join(cmd_args)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("Registration to template space failed")

        # Compute SRR in template space
        dir_input_mc = os.path.join(
            dir_output_recon_template_space, "motion_correction")
        cmd_args = ["niftymic_reconstruct_volume_from_slices"]
        cmd_args.append("--filenames %s" % (" ").join(filenames))
        cmd_args.append("--dir-input-mc '%s'" % dir_input_mc)
        cmd_args.append("--output '%s'" % srr_template)
        cmd_args.append("--reconstruction-space '%s'" % template)
        cmd_args.append("--iter-max %d" % args.iter_max)
        cmd_args.append("--alpha %s" % args.alpha)
        cmd_args.append("--suffix-mask '%s'" % args.suffix_mask)
        cmd_args.append("--verbose %s" % args.verbose)
        if args.slice_thicknesses is not None:
            cmd_args.append("--slice-thicknesses %s" %
                            " ".join(map(str, args.slice_thicknesses)))
        if args.sda:
            cmd_args.append("--sda")

        cmd = (" ").join(cmd_args)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("Reconstruction in template space failed")

        # Compute SRR mask in template space
        if 1:
            dir_motion_correction = os.path.join(
                dir_output_recon_template_space, "motion_correction")
            cmd_args = ["niftymic_reconstruct_volume_from_slices"]
            cmd_args.append("--filenames %s" % " ".join(filenames_masks))
            cmd_args.append("--dir-input-mc '%s'" % dir_motion_correction)
            cmd_args.append("--output '%s'" % srr_template_mask)
            cmd_args.append("--reconstruction-space '%s'" % srr_template)
            cmd_args.append("--suffix-mask '%s'" % args.suffix_mask)
            cmd_args.append("--mask")
            if args.slice_thicknesses is not None:
                cmd_args.append("--slice-thicknesses %s" %
                                " ".join(map(str, args.slice_thicknesses)))
            if args.sda:
                cmd_args.append("--sda")
                cmd_args.append("--alpha 1")
            else:
                cmd_args.append("--alpha 0.1")
                cmd_args.append("--iter-max 5")
            cmd = (" ").join(cmd_args)
            ph.execute_command(cmd)

        # Copy SRR to output directory
        output = "%sSRR_Stacks%d.nii.gz" % (
            args.prefix_output, len(args.filenames))
        path_to_output = os.path.join(args.dir_output, output)
        cmd = "cp -p %s %s" % (srr_template, path_to_output)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("Copy of SRR to output directory failed")

        # Multiply template mask with reconstruction
        cmd_args = ["niftymic_multiply"]
        fnames = [
            srr_template,
            srr_template_mask,
        ]
        output_masked = "Masked_%s" % output
        path_to_output_masked = os.path.join(args.dir_output, output_masked)
        cmd_args.append("--filenames %s" % " ".join(fnames))
        cmd_args.append("--output '%s'" % path_to_output_masked)
        cmd = (" ").join(cmd_args)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("SRR brain masking failed")

    else:
        elapsed_time_template = ph.get_zero_time()

    if args.run_diagnostics:

        dir_input_mc = os.path.join(
            dir_output_recon_template_space, "motion_correction")
        dir_output_orig_vs_proj = os.path.join(
            dir_output_diagnostics, "original_vs_projected")
        dir_output_selfsimilarity = os.path.join(
            dir_output_diagnostics, "selfsimilarity")
        dir_output_orig_vs_proj_pdf = os.path.join(
            dir_output_orig_vs_proj, "pdf")

        # Show slice coverage over reconstruction space
        exe = os.path.abspath(show_slice_coverage.__file__)
        cmd_args = ["python %s" % exe]
        cmd_args.append("--filenames %s" % (" ").join(filenames))
        cmd_args.append("--dir-input-mc '%s'" % dir_input_mc)
        cmd_args.append("--reconstruction-space '%s'" % srr_template)
        cmd_args.append("--output '%s'" % srr_slice_coverage)
        cmd = (" ").join(cmd_args)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("Slice coverage visualization failed")

        # Get simulated/projected slices
        exe = os.path.abspath(simulate_stacks_from_reconstruction.__file__)
        cmd_args = ["python %s" % exe]
        cmd_args.append("--filenames %s" % (" ").join(filenames))
        if args.filenames_masks is not None:
            cmd_args.append("--filenames-masks %s" %
                            (" ").join(filenames_masks))
        cmd_args.append("--dir-input-mc '%s'" % dir_input_mc)
        cmd_args.append("--dir-output '%s'" % dir_output_orig_vs_proj)
        cmd_args.append("--reconstruction '%s'" % srr_template)
        cmd_args.append("--copy-data 1")
        if args.slice_thicknesses is not None:
            cmd_args.append("--slice-thicknesses %s" %
                            " ".join(map(str, args.slice_thicknesses)))
        # cmd_args.append("--verbose %s" % args.verbose)
        cmd = (" ").join(cmd_args)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("SRR slice projections failed")

        filenames_simulated = [
            os.path.join(dir_output_orig_vs_proj, os.path.basename(f))
            for f in filenames]

        # Evaluate slice similarities to ground truth
        exe = os.path.abspath(evaluate_simulated_stack_similarity.__file__)
        cmd_args = ["python %s" % exe]
        cmd_args.append("--filenames %s" % (" ").join(filenames_simulated))
        if args.filenames_masks is not None:
            cmd_args.append("--filenames-masks %s" %
                            (" ").join(filenames_masks))
        cmd_args.append("--measures NCC SSIM")
        cmd_args.append("--dir-output '%s'" % dir_output_selfsimilarity)
        cmd = (" ").join(cmd_args)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("Evaluation of slice similarities failed")

        # Generate figures showing the quantitative comparison
        exe = os.path.abspath(
            show_evaluated_simulated_stack_similarity.__file__)
        cmd_args = ["python %s" % exe]
        cmd_args.append("--dir-input %s" % dir_output_selfsimilarity)
        cmd_args.append("--dir-output '%s'" % dir_output_selfsimilarity)
        cmd = (" ").join(cmd_args)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            ph.print_warning("Visualization of slice similarities failed")

        # Generate pdfs showing all the side-by-side comparisons
        if 0:
            exe = os.path.abspath(
                export_side_by_side_simulated_vs_original_slice_comparison.__file__)
            cmd_args = ["python %s" % exe]
            cmd_args.append("--filenames %s" % (" ").join(filenames_simulated))
            cmd_args.append("--dir-output '%s'" % dir_output_orig_vs_proj_pdf)
            cmd = "python %s %s" % (exe, (" ").join(cmd_args))
            cmd = (" ").join(cmd_args)
            exit_code = ph.execute_command(cmd)
            if exit_code != 0:
                raise RuntimeError("Generation of PDF overview failed")

    ph.print_title("Summary")
    print("Computational Time for Bias Field Correction: %s" %
          elapsed_time_bias)
    print("Computational Time for Volumetric Reconstruction: %s" %
          elapsed_time_volrec)
    print("Computational Time for Pipeline: %s" %
          ph.stop_timing(time_start))

    return 0


if __name__ == '__main__':
    main()
