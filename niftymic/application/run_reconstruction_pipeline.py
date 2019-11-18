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

    time_start_total = ph.start_timing()

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
    input_parser.add_suffix_mask(default="")
    input_parser.add_dir_output(required=True)
    input_parser.add_alpha(default=0.01)
    input_parser.add_verbose(default=0)
    input_parser.add_prefix_output(default="srr_")
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
        option_string="--template",
        type=str,
        required=False,
        help="Template image used for template space alignment and to define "
        "the reconstruction space. "
        "If not given, it is automatically estimated using the fetal brain "
        "atlas",
    )
    input_parser.add_option(
        option_string="--template-mask",
        type=str,
        required=False,
        help="Template image mask. "
        "Must be given in case template is specified.",
    )
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
    input_parser.add_threshold_first(default=0.5)
    input_parser.add_threshold(default=0.8)
    input_parser.add_argument(
        "--sda", "-sda",
        action='store_true',
        help="If given, the volume is reconstructed using "
        "Scattered Data Approximation (Vercauteren et al., 2006). "
        "--alpha is considered the value for the standard deviation then. "
        "Recommended value is, e.g., --alpha 0.8"
    )
    input_parser.add_argument(
        "--v2v-robust", "-v2v-robust",
        action='store_true',
        help="If given, a more robust volume-to-volume registration step is "
        "performed, i.e. four rigid registrations are performed using four "
        "rigid transform initializations based on "
        "principal component alignment of associated masks."
    )
    input_parser.add_interleave(default=3)
    input_parser.add_argument(
        "--s2v-hierarchical", "-s2v-hierarchical",
        action='store_true',
        help="If given, a hierarchical approach for the first slice-to-volume "
        "registration cycle is used, i.e. sub-packages defined by the "
        "specified interleave (--interleave) are registered until each "
        "slice is registered independently."
    )

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    if args.template is not None:
        if args.template_mask is None:
            raise ValueError(
                "If template image is given, also its mask needs to be "
                "provided")

    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))

    dir_output_preprocessing = os.path.join(
        args.dir_output, "preprocessing_n4itk")
    dir_output_recon_subject_space = os.path.join(
        args.dir_output, "recon_subject_space")
    dir_output_recon_template_space = os.path.join(
        args.dir_output, "recon_template_space")
    dir_output_diagnostics = os.path.join(
        args.dir_output, "diagnostics")

    srr_subject = os.path.join(
        dir_output_recon_subject_space,
        "%ssubject.nii.gz" % args.prefix_output)
    srr_subject_mask = ph.append_to_filename(srr_subject, "_mask")
    srr_template = os.path.join(
        dir_output_recon_template_space,
        "%stemplate.nii.gz" % args.prefix_output)
    srr_template_mask = ph.append_to_filename(srr_template, "_mask")
    trafo_template = os.path.join(
        dir_output_recon_template_space,
        "%stemplate_transform_sitk.txt" % args.prefix_output)
    srr_slice_coverage = os.path.join(
        dir_output_diagnostics,
        "%stemplate_slicecoverage.nii.gz" % args.prefix_output)

    if args.bias_field_correction and args.run_bias_field_correction:
        time_start = ph.start_timing()
        for i, f in enumerate(args.filenames):
            output = os.path.join(
                dir_output_preprocessing, os.path.basename(f))
            cmd_args = []
            cmd_args.append("--filename '%s'" % f)
            cmd_args.append("--filename-mask '%s'" % args.filenames_masks[i])
            cmd_args.append("--output '%s'" % output)
            # cmd_args.append("--verbose %d" % args.verbose)
            cmd_args.append("--log-config %d" % args.log_config)
            cmd = "niftymic_correct_bias_field %s" % (" ").join(cmd_args)
            exit_code = ph.execute_command(cmd)
            if exit_code != 0:
                raise RuntimeError("Bias field correction failed")
        elapsed_time_bias = ph.stop_timing(time_start)
        filenames = [os.path.join(dir_output_preprocessing, os.path.basename(f))
                     for f in args.filenames]
    elif args.bias_field_correction and not args.run_bias_field_correction:
        elapsed_time_bias = ph.get_zero_time()
        filenames = [os.path.join(dir_output_preprocessing, os.path.basename(f))
                     for f in args.filenames]
    else:
        elapsed_time_bias = ph.get_zero_time()
        filenames = args.filenames

    # Specify target stack for intensity correction and reconstruction space
    if args.target_stack is None:
        target_stack = filenames[0]
    else:
        try:
            target_stack_index = args.filenames.index(args.target_stack)
        except ValueError as e:
            raise ValueError(
                "--target-stack must correspond to an image as provided by "
                "--filenames")
        target_stack = filenames[target_stack_index]

    # Add single quotes around individual filenames to account for whitespaces
    filenames = ["'" + f + "'" for f in filenames]
    filenames_masks = ["'" + f + "'" for f in args.filenames_masks]

    if args.run_recon_subject_space:
        time_start = ph.start_timing()

        cmd_args = ["niftymic_reconstruct_volume"]
        cmd_args.append("--filenames %s" % (" ").join(filenames))
        cmd_args.append("--filenames-masks %s" % (" ").join(filenames_masks))
        cmd_args.append("--multiresolution %d" % args.multiresolution)
        cmd_args.append("--target-stack '%s'" % target_stack)
        cmd_args.append("--output '%s'" % srr_subject)
        cmd_args.append("--suffix-mask '%s'" % args.suffix_mask)
        cmd_args.append("--intensity-correction %d" %
                        args.intensity_correction)
        cmd_args.append("--alpha %s" % args.alpha)
        cmd_args.append("--iter-max %d" % args.iter_max)
        cmd_args.append("--two-step-cycles %d" % args.two_step_cycles)
        cmd_args.append("--outlier-rejection %d" % args.outlier_rejection)
        cmd_args.append("--threshold-first %f" % args.threshold_first)
        cmd_args.append("--threshold %f" % args.threshold)
        if args.slice_thicknesses is not None:
            cmd_args.append("--slice-thicknesses %s" %
                            " ".join(map(str, args.slice_thicknesses)))
        cmd_args.append("--verbose %d" % args.verbose)
        cmd_args.append("--log-config %d" % args.log_config)
        if args.isotropic_resolution is not None:
            cmd_args.append("--isotropic-resolution %f" %
                            args.isotropic_resolution)
        if args.reference is not None:
            cmd_args.append("--reference '%s'" % args.reference)
        if args.reference_mask is not None:
            cmd_args.append("--reference-mask '%s'" % args.reference_mask)
        if args.sda:
            cmd_args.append("--sda")
        if args.v2v_robust:
            cmd_args.append("--v2v-robust")
        if args.s2v_hierarchical:
            cmd_args.append("--s2v-hierarchical")

        cmd = (" ").join(cmd_args)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("Reconstruction in subject space failed")
        elapsed_time_recon_subject_space = ph.stop_timing(time_start)

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
            cmd_args.append("--log-config %d" % args.log_config)
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

    else:
        elapsed_time_recon_subject_space = ph.get_zero_time()

    if args.run_recon_template_space:
        time_start = ph.start_timing()

        if args.template is not None:
            template = args.template
            template_mask = args.template_mask
        else:
            template_stack_estimator = \
                tse.TemplateStackEstimator.from_mask(srr_subject_mask)
            gestational_age = template_stack_estimator.get_estimated_gw()
            ph.print_info("Estimated gestational age: %d" % gestational_age)

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
        cmd_args.append("--log-config %d" % args.log_config)
        if args.initial_transform is None:
            cmd_args.append("--init-pca")
        else:
            cmd_args.append(
                "--initial-transform '%s'" % args.initial_transform)
        cmd = (" ").join(cmd_args)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("Registration to template space failed")
        elapsed_time_register_image = ph.stop_timing(time_start)
        time_start = ph.start_timing()

        # Compute SRR in template space
        dir_input_mc = os.path.join(
            dir_output_recon_template_space, "motion_correction")
        cmd_args = ["niftymic_reconstruct_volume_from_slices"]
        cmd_args.append("--filenames %s" % (" ").join(filenames))
        cmd_args.append("--filenames-masks %s" % (" ").join(filenames_masks))
        cmd_args.append("--dir-input-mc '%s'" % dir_input_mc)
        cmd_args.append("--output '%s'" % srr_template)
        cmd_args.append("--reconstruction-space '%s'" % template)
        cmd_args.append("--target-stack '%s'" % target_stack)
        cmd_args.append("--iter-max %d" % args.iter_max)
        cmd_args.append("--alpha %s" % args.alpha)
        cmd_args.append("--suffix-mask '%s'" % args.suffix_mask)
        cmd_args.append("--verbose %s" % args.verbose)
        cmd_args.append("--log-config %d" % args.log_config)
        if args.slice_thicknesses is not None:
            cmd_args.append("--slice-thicknesses %s" %
                            " ".join(map(str, args.slice_thicknesses)))
        if args.sda:
            cmd_args.append("--sda")
        cmd = (" ").join(cmd_args)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("Reconstruction in template space failed")
        elapsed_time_recon_template_space = ph.stop_timing(time_start)

        # Compute mask in template space
        if 1:
            time_start = ph.start_timing()
            dir_motion_correction = os.path.join(
                dir_output_recon_template_space, "motion_correction")
            cmd_args = ["niftymic_reconstruct_volume_from_slices"]
            cmd_args.append("--filenames %s" % " ".join(filenames_masks))
            cmd_args.append("--dir-input-mc '%s'" % dir_motion_correction)
            cmd_args.append("--output '%s'" % srr_template_mask)
            cmd_args.append("--reconstruction-space '%s'" % srr_template)
            cmd_args.append("--suffix-mask '%s'" % args.suffix_mask)
            cmd_args.append("--log-config %d" % args.log_config)
            cmd_args.append("--mask")
            if args.slice_thicknesses is not None:
                cmd_args.append("--slice-thicknesses %s" %
                                " ".join(map(str, args.slice_thicknesses)))

            # SRR approach
            # cmd_args.append("--alpha 0.1")
            # cmd_args.append("--iter-max 5")

            # SDA much faster than SRR and visually barely different for mask
            cmd_args.append("--sda")
            cmd_args.append("--alpha 1")

            cmd = (" ").join(cmd_args)
            ph.execute_command(cmd)
            elapsed_time_recon_template_space_mask = ph.stop_timing(
                time_start)

        # Copy SRR to output directory
        if 0:
            output = "%sSRR_Stacks%d.nii.gz" % (
                args.prefix_output, len(args.filenames))
            path_to_output = os.path.join(args.dir_output, output)
            cmd = "cp -p '%s' '%s'" % (srr_template, path_to_output)
            exit_code = ph.execute_command(cmd)
            if exit_code != 0:
                raise RuntimeError("Copy of SRR to output directory failed")

        # Multiply template mask with reconstruction
        if 0:
            cmd_args = ["niftymic_multiply"]
            fnames = [
                srr_template,
                srr_template_mask,
            ]
            output_masked = "Masked_%s" % output
            path_to_output_masked = os.path.join(
                args.dir_output, output_masked)
            cmd_args.append("--filenames %s" % " ".join(fnames))
            cmd_args.append("--output '%s'" % path_to_output_masked)
            cmd = (" ").join(cmd_args)
            exit_code = ph.execute_command(cmd)
            if exit_code != 0:
                raise RuntimeError("SRR brain masking failed")

    else:
        elapsed_time_register_image = ph.get_zero_time()
        elapsed_time_recon_template_space = ph.get_zero_time()
        elapsed_time_recon_template_space_mask = ph.get_zero_time()

    if args.run_diagnostics:
        time_start = ph.start_timing()

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
            "'%s" % os.path.join(dir_output_orig_vs_proj, os.path.basename(f))
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
            raise RuntimeError("Evaluation of stack similarities failed")

        # Generate figures showing the quantitative comparison
        exe = os.path.abspath(
            show_evaluated_simulated_stack_similarity.__file__)
        cmd_args = ["python %s" % exe]
        cmd_args.append("--dir-input '%s'" % dir_output_selfsimilarity)
        cmd_args.append("--dir-output '%s'" % dir_output_selfsimilarity)
        cmd = (" ").join(cmd_args)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            ph.print_warning("Visualization of stack similarities failed")

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
        elapsed_time_diagnostics = ph.stop_timing(time_start)

    ph.print_title("Summary")
    exe_file_info = os.path.basename(os.path.abspath(__file__)).split(".")[0]
    print("%s | Computational Time for Bias Field Corrections: %s" % (
          exe_file_info, elapsed_time_bias))
    print("%s | Computational Time for Subject Space Reconstruction: %s" % (
          exe_file_info, elapsed_time_recon_subject_space))
    print("%s | Computational Time for Template Space Alignment: %s" % (
          exe_file_info, elapsed_time_register_image))
    print("%s | Computational Time for Template Space Reconstruction: %s" % (
          exe_file_info, elapsed_time_recon_template_space))
    print("%s | Computational Time for Template Space Reconstruction (Mask): %s" % (
          exe_file_info, elapsed_time_recon_template_space_mask))
    if args.run_diagnostics:
        print("%s | Computational Time for Diagnostics: %s" % (
              exe_file_info, elapsed_time_diagnostics))
    print("%s | Computational Time for Pipeline: %s" % (
          exe_file_info, ph.stop_timing(time_start_total)))

    return 0


if __name__ == '__main__':
    main()
