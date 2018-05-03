##
# \file run_reconstruction_pipeline.py
# \brief      Script to execute entire reconstruction pipeline
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       October 2017
#

import numpy as np
import os
import re

import niftymic.validation.simulate_stacks_from_reconstruction as \
    simulate_stacks_from_reconstruction
import niftymic.validation.evaluate_simulated_stack_similarity as \
    evaluate_simulated_stack_similarity
import niftymic.validation.show_evaluated_simulated_stack_similarity as \
    show_evaluated_simulated_stack_similarity
import niftymic.validation.export_side_by_side_simulated_vs_original_slice_comparison as \
    export_side_by_side_simulated_vs_original_slice_comparison
import pysitk.python_helper as ph
from niftymic.definitions import DIR_TEMPLATES
from niftymic.utilities.input_arparser import InputArgparser


def main():

    time_start = ph.start_timing()

    np.set_printoptions(precision=3)

    input_parser = InputArgparser(
        description="Run reconstruction pipeline including "
        "(i) preprocessing (bias field correction + intensity correction), "
        "(ii) volumetric reconstruction in subject space, "
        "and (iii) volumetric reconstruction in template space.",
    )
    input_parser.add_filenames(required=True)
    input_parser.add_target_stack(required=False)
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_dir_output(required=True)
    input_parser.add_alpha(default=0.01)
    input_parser.add_verbose(default=0)
    input_parser.add_gestational_age(required=False)
    input_parser.add_prefix_output(default="")
    input_parser.add_search_angle(default=180)
    input_parser.add_multiresolution(default=0)
    input_parser.add_log_script_execution(default=1)
    input_parser.add_dir_input_templates(default=DIR_TEMPLATES)
    input_parser.add_isotropic_resolution()
    input_parser.add_reference()
    input_parser.add_reference_mask()
    input_parser.add_bias_field_correction(default=1)
    input_parser.add_intensity_correction(default=1)
    input_parser.add_iter_max(default=10)
    input_parser.add_two_step_cycles(default=3)
    input_parser.add_option(
        option_string="--run-recon-subject-space",
        type=int,
        help="Turn on/off reconstruction in subject space",
        default=1)
    input_parser.add_option(
        option_string="--run-recon-template-space",
        type=int,
        help="Turn on/off reconstruction in template space",
        default=1)
    input_parser.add_option(
        option_string="--run-data-vs-simulated-data",
        type=int,
        help="Turn on/off comparison of data vs data simulated from the "
        "obtained volumetric reconstruction",
        default=1)
    input_parser.add_outlier_rejection(default=1)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # Write script execution call
    if args.log_script_execution:
        input_parser.write_performed_script_execution(
            os.path.abspath(__file__))

    prefix_bias = "N4ITK_"
    prefix_ic = "IC_"
    dir_output_preprocessing = os.path.join(
        args.dir_output, "preprocessing")
    dir_output_recon_subject_space = os.path.join(
        args.dir_output, "recon_subject_space")
    dir_output_recon_template_space = os.path.join(
        args.dir_output, "recon_template_space")
    dir_output_data_vs_simulatd_data = os.path.join(
        args.dir_output, "data_vs_simulated_data")

    if args.run_recon_template_space and args.gestational_age is None:
        raise IOError("Gestational age must be set in order to pick the "
                      "right template")

    if args.target_stack is None:
        target_stack = args.filenames[0]
    else:
        target_stack = args.target_stack

    if args.bias_field_correction:
        cmd_args = []
        cmd_args.append("--filenames %s" % (" ").join(args.filenames))
        cmd_args.append("--dir-output %s" % dir_output_preprocessing)
        cmd_args.append("--prefix-output %s" % prefix_bias)
        cmd_args.append("--suffix-mask %s" % args.suffix_mask)
        # cmd_args.append("--verbose %d" % args.verbose)
        cmd = "niftymic_correct_bias_field %s" % (" ").join(cmd_args)
        time_start_bias = ph.start_timing()
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("Bias field correction failed")
        elapsed_time_bias = ph.stop_timing(time_start_bias)
        filenames = [os.path.join(dir_output_preprocessing, "%s%s" % (
            prefix_bias, os.path.basename(f)))
            for f in args.filenames]
    else:
        elapsed_time_bias = ph.get_zero_time()
        filenames = args.filenames

    if args.run_recon_subject_space:

        target_stack_index = args.filenames.index(target_stack)

        cmd_args = []
        cmd_args.append("--filenames %s" % (" ").join(filenames))
        cmd_args.append("--multiresolution %d" % args.multiresolution)
        cmd_args.append("--target-stack-index %d" % target_stack_index)
        cmd_args.append("--dir-output %s" % dir_output_recon_subject_space)
        cmd_args.append("--suffix-mask %s" % args.suffix_mask)
        cmd_args.append("--intensity-correction %d" %
                        args.intensity_correction)
        cmd_args.append("--alpha %s" % args.alpha)
        cmd_args.append("--iter-max %d" % args.iter_max)
        cmd_args.append("--two-step-cycles %d" % args.two_step_cycles)
        cmd_args.append("--outlier-rejection %d" %
                        args.outlier_rejection)
        cmd_args.append("--verbose %d" % args.verbose)
        if args.isotropic_resolution is not None:
            cmd_args.append("--isotropic-resolution %f" %
                            args.isotropic_resolution)
        if args.reference is not None:
            cmd_args.append("--reference %s" % args.reference)
        if args.reference_mask is not None:
            cmd_args.append("--reference-mask %s" % args.reference_mask)
        cmd = "niftymic_reconstruct_volume %s" % (" ").join(cmd_args)
        time_start_volrec = ph.start_timing()
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("Reconstruction in subject space failed")
        elapsed_time_volrec = ph.stop_timing(time_start_volrec)
    else:
        elapsed_time_volrec = ph.get_zero_time()

    if args.run_recon_template_space:
        # register recon to template space
        template = os.path.join(
            args.dir_input_templates,
            "STA%d.nii.gz" % args.gestational_age)
        template_mask = os.path.join(
            args.dir_input_templates,
            "STA%d_mask.nii.gz" % args.gestational_age)
        pattern = "[a-zA-Z0-9_]+(stacks)[a-zA-Z0-9_]+(.nii.gz)"
        p = re.compile(pattern)
        reconstruction = [
            os.path.join(
                dir_output_recon_subject_space, p.match(f).group(0))
            for f in os.listdir(dir_output_recon_subject_space)
            if p.match(f)][0]
        cmd_args = []
        cmd_args.append("--moving %s" % reconstruction)
        cmd_args.append("--fixed %s" % template)
        # cmd_args.append("--template-mask %s" % template_mask)
        cmd_args.append("--dir-input %s" % os.path.join(
            dir_output_recon_subject_space,
            "motion_correction"))
        cmd_args.append("--dir-output %s" % dir_output_recon_template_space)
        cmd_args.append("--suffix-mask %s" % args.suffix_mask)
        cmd_args.append("--verbose %s" % args.verbose)
        cmd = "niftymic_register_image %s" % (" ").join(cmd_args)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("Registration to template space failed")

        # reconstruct volume in template space
        # pattern = "[a-zA-Z0-9_.]+(ResamplingToTemplateSpace.nii.gz)"
        # p = re.compile(pattern)
        # reconstruction_space = [
        #     os.path.join(dir_output_recon_template_space, p.match(f).group(0))
        #     for f in os.listdir(dir_output_recon_template_space)
        #     if p.match(f)][0]

        dir_input = os.path.join(
            dir_output_recon_template_space, "motion_correction")
        cmd_args = []
        cmd_args.append("--dir-input %s" % dir_input)
        cmd_args.append("--dir-output %s" % dir_output_recon_template_space)
        cmd_args.append("--reconstruction-space %s" % template)
        cmd_args.append("--iter-max %d" % args.iter_max)
        cmd_args.append("--alpha %s" % args.alpha)
        cmd_args.append("--suffix-mask %s" % args.suffix_mask)

        cmd = "niftymic_reconstruct_volume_from_slices %s" % \
            (" ").join(cmd_args)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("Reconstruction in template space failed")

        pattern = "[a-zA-Z0-9_.]+(stacks[0-9]+).*(.nii.gz)"
        p = re.compile(pattern)
        reconstruction = {
            p.match(f).group(1):
            os.path.join(
                dir_output_recon_template_space, p.match(f).group(0))
            for f in os.listdir(dir_output_recon_template_space)
            if p.match(f) and not p.match(f).group(0).endswith(
                "ResamplingToTemplateSpace.nii.gz")}
        key = reconstruction.keys()[0]
        path_to_recon = reconstruction[key]

        # Copy SRR to output directory
        output = "%sSRR_%s_GW%d.nii.gz" % (
            args.prefix_output, key, args.gestational_age)
        path_to_output = os.path.join(args.dir_output, output)
        cmd = "cp -p %s %s" % (path_to_recon, path_to_output)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("Copy of SRR to output directory failed")

        # Multiply template mask with reconstruction
        cmd_args = []
        cmd_args.append("--filename %s" % path_to_output)
        cmd_args.append("--gestational-age %s" % args.gestational_age)
        cmd_args.append("--verbose %s" % args.verbose)
        cmd_args.append("--dir-input-templates %s " % args.dir_input_templates)
        cmd = "niftymic_multiply_stack_with_mask %s" % (
            " ").join(cmd_args)
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("SRR brain masking failed")

    else:
        elapsed_time_template = ph.get_zero_time()

    if args.run_data_vs_simulated_data:

        dir_input = os.path.join(
            dir_output_recon_template_space, "motion_correction")

        pattern = "[a-zA-Z0-9_.]+(stacks[0-9]+).*(.nii.gz)"
        # pattern = "Masked_[a-zA-Z0-9_.]+(stacks[0-9]+).*(.nii.gz)"
        p = re.compile(pattern)
        reconstruction = {
            p.match(f).group(1):
            os.path.join(
                dir_output_recon_template_space, p.match(f).group(0))
            for f in os.listdir(dir_output_recon_template_space)
            if p.match(f) and not p.match(f).group(0).endswith(
                "ResamplingToTemplateSpace.nii.gz")}
        key = reconstruction.keys()[0]
        path_to_recon = reconstruction[key]

        # Get simulated/projected slices
        cmd_args = []
        cmd_args.append("--dir-input %s" % dir_input)
        cmd_args.append("--dir-output %s" % dir_output_data_vs_simulatd_data)
        cmd_args.append("--reconstruction %s" % path_to_recon)
        cmd_args.append("--copy-data 1")
        cmd_args.append("--suffix-mask %s" % args.suffix_mask)
        # cmd_args.append("--verbose %s" % args.verbose)
        exe = os.path.abspath(simulate_stacks_from_reconstruction.__file__)
        cmd = "python %s %s" % (exe, (" ").join(cmd_args))
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("SRR slice projections failed")

        filenames_simulated = [
            os.path.join(dir_output_data_vs_simulatd_data, os.path.basename(f))
            for f in filenames]

        dir_output_evaluation = os.path.join(
            dir_output_data_vs_simulatd_data, "evaluation")
        dir_output_figures = os.path.join(
            dir_output_data_vs_simulatd_data, "figures")
        dir_output_side_by_side = os.path.join(
            dir_output_figures, "side-by-side")

        # Evaluate slice similarities to ground truth
        cmd_args = []
        cmd_args.append("--filenames %s" % (" ").join(filenames_simulated))
        cmd_args.append("--suffix-mask %s" % args.suffix_mask)
        cmd_args.append("--measures NCC SSIM")
        cmd_args.append("--dir-output %s" % dir_output_evaluation)
        exe = os.path.abspath(evaluate_simulated_stack_similarity.__file__)
        cmd = "python %s %s" % (exe, (" ").join(cmd_args))
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            raise RuntimeError("Evaluation of slice similarities failed")

        # Generate figures showing the quantitative comparison
        cmd_args = []
        cmd_args.append("--dir-input %s" % dir_output_evaluation)
        cmd_args.append("--dir-output %s" % dir_output_figures)
        exe = os.path.abspath(
            show_evaluated_simulated_stack_similarity.__file__)
        cmd = "python %s %s" % (exe, (" ").join(cmd_args))
        exit_code = ph.execute_command(cmd)
        if exit_code != 0:
            ph.print_warning("Visualization of slice similarities failed")

        # Generate pdfs showing all the side-by-side comparisons
        cmd_args = []
        cmd_args.append("--filenames %s" % (" ").join(filenames_simulated))
        cmd_args.append("--dir-output %s" % dir_output_side_by_side)
        exe = os.path.abspath(
            export_side_by_side_simulated_vs_original_slice_comparison.__file__)
        cmd = "python %s %s" % (exe, (" ").join(cmd_args))
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
