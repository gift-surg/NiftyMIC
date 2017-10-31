##
# \file run_reconstruction_pipeline.py
# \brief      Script to execute entire reconstruction pipeline
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       October 2017
#

# Import libraries
import SimpleITK as sitk
import argparse
import numpy as np
import sys
import os
import re

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

# Import modules
import niftymic.base.data_reader as dr
import niftymic.base.stack as st
from niftymic.utilities.input_arparser import InputArgparser
from niftymic.definitions import DIR_TEMPLATES


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
    input_parser.add_target_stack(required=True)
    input_parser.add_dir_output(required=True)
    input_parser.add_verbose(default=0)
    input_parser.add_option(
        option_string="--registration",
        type=int,
        help="Turn on/off registration from image to reference prior to "
        "intensity correction.",
        default=1)
    input_parser.add_option(
        option_string="--run-preprocessing",
        type=int,
        help="Turn on/off preprocessing including bias field and linear "
        "intensity correction",
        default=1)
    input_parser.add_option(
        option_string="--run-recon-subject-space",
        type=int,
        help="Turn on/off reconstruction",
        default=1)
    input_parser.add_option(
        option_string="--run-recon-template-space",
        type=int,
        help="Turn on/off registration to template",
        default=1)
    input_parser.add_option(
        option_string="--gestational-age",
        type=int,
        help="Gestational age of fetus to reconstruct",
        required=False,
        default=None)
    input_parser.add_option(
        option_string="--dir-input-templates",
        type=str,
        help="Input directory for templates",
        default=DIR_TEMPLATES)
    input_parser.add_prefix_output(default="")
    input_parser.add_search_angle(default=180)
    input_parser.add_multiresolution(default=0)
    input_parser.add_log_script_execution(default=1)

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

    if args.run_recon_template_space and args.gestational_age is None:
        raise IOError("Gestational age must be set to define template")

    if args.run_preprocessing:

        # run bias field correction
        filenames = list(args.filenames)
        cmd_args = []
        cmd_args.append("--filenames %s" % (" ").join(filenames))
        cmd_args.append("--dir-output %s" % dir_output_preprocessing)
        cmd_args.append("--prefix-output %s" % prefix_bias)
        # cmd_args.append("--verbose %d" % args.verbose)
        cmd = "niftymic_correct_bias_field %s" % (" ").join(cmd_args)
        time_start_bias = ph.start_timing()
        ph.execute_command(cmd)
        elapsed_time_bias = ph.stop_timing(time_start_bias)

        # run intensity correction
        filenames = [os.path.join(dir_output_preprocessing, "%s%s" % (
            prefix_bias, os.path.basename(f))) for f in filenames]
        target_stack = os.path.join(dir_output_preprocessing, "%s%s" % (
            prefix_bias, os.path.basename(args.target_stack)))

        cmd_args = []
        cmd_args.append("--filenames %s" % (" ").join(filenames))
        cmd_args.append("--reference %s" % target_stack)
        cmd_args.append("--registration %d" % args.registration)
        cmd_args.append("--search-angle %d" % args.search_angle)
        cmd_args.append("--dir-output %s" % dir_output_preprocessing)
        cmd_args.append("--prefix-output %s" % prefix_ic)
        # cmd_args.append("--verbose %d" % args.verbose)
        cmd = "niftymic_correct_intensities %s" % (" ").join(cmd_args)
        time_start_ic = ph.start_timing()
        ph.execute_command(cmd)
        elapsed_time_ic = ph.stop_timing(time_start_ic)
    else:
        elapsed_time_bias = ph.get_zero_time()
        elapsed_time_ic = ph.get_zero_time()

    if args.run_recon_subject_space:

        # reconstruct volume in subject space
        filenames = [os.path.join(dir_output_preprocessing, "%s%s%s" % (
            prefix_ic, prefix_bias, os.path.basename(f)))
            for f in args.filenames]
        target_stack_index = args.filenames.index(args.target_stack)

        cmd_args = []
        cmd_args.append("--filenames %s" % (" ").join(filenames))
        cmd_args.append("--multiresolution %d" % args.multiresolution)
        cmd_args.append("--target-stack-index %d" % target_stack_index)
        cmd_args.append("--dir-output %s" % dir_output_recon_subject_space)
        cmd_args.append("--verbose %d" % args.verbose)
        cmd = "niftymic_reconstruct_volume %s" % (" ").join(cmd_args)
        time_start_volrec = ph.start_timing()
        ph.execute_command(cmd)
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
        cmd_args.append("--reconstruction %s" % reconstruction)
        cmd_args.append("--template %s" % template)
        # cmd_args.append("--template-mask %s" % template_mask)
        cmd_args.append("--dir-input %s" % os.path.join(
            dir_output_recon_subject_space,
            "motion_correction"))
        cmd_args.append("--dir-output %s" % dir_output_recon_template_space)
        cmd_args.append("--verbose %s" % args.verbose)
        cmd = "niftymic_register_to_template %s" % (" ").join(cmd_args)
        ph.execute_command(cmd)

        # reconstruct volume in template space
        pattern = "[a-zA-Z0-9_.]+(ResamplingToTemplateSpace.nii.gz)"
        p = re.compile(pattern)
        reconstruction_space = [
            os.path.join(dir_output_recon_template_space, p.match(f).group(0))
            for f in os.listdir(dir_output_recon_template_space)
            if p.match(f)][0]

        dir_input = os.path.join(
            dir_output_recon_template_space, "motion_correction")
        cmd_args = []
        cmd_args.append("--dir-input %s" % dir_input)
        cmd_args.append("--dir-output %s" % dir_output_recon_template_space)
        cmd_args.append("--reconstruction-space %s" % reconstruction_space)
        cmd = "niftymic_reconstruct_volume_from_slices %s" % \
            (" ").join(cmd_args)
        ph.execute_command(cmd)

        # Copy SRR to output directory
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
        output = "%sSRR_%s.nii.gz" % (args.prefix_output, key)
        path_to_recon = os.path.join(
            dir_output_recon_template_space, reconstruction[key])
        path_to_output = os.path.join(args.dir_output, output)
        cmd = "cp %s %s" % (path_to_recon, path_to_output)
        ph.execute_command(cmd)

        # Apply masking of template
        cmd_args = []
        cmd_args.append("--filename %s" % path_to_output)
        cmd_args.append("--gestational-age %s" % args.gestational_age)
        cmd_args.append("--verbose %s" % args.verbose)
        cmd = "niftymic_multiply_stack_with_mask %s" % (
            " ").join(cmd_args)
        ph.execute_command(cmd)

    else:
        elapsed_time_template = ph.get_zero_time()

    ph.print_title("Summary")
    print("Computational Time for Bias Field Correction: %s" %
          elapsed_time_bias)
    print("Computational Time for Intensity Correction: %s" %
          elapsed_time_ic)
    print("Computational Time for Volumetric Reconstruction: %s" %
          elapsed_time_volrec)
    print("Computational Time for Pipeline: %s" %
          ph.stop_timing(time_start))

    return 0


if __name__ == '__main__':
    main()
