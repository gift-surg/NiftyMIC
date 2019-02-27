##
# \file register_image.py
# \brief      Script to register the obtained reconstruction to a template
#             space.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       October 2017
#

import re
import os
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.stack as st
import niftymic.base.data_reader as dr
import niftymic.registration.niftyreg as niftyreg
import niftymic.registration.transform_initializer as tinit
from niftymic.utilities.input_arparser import InputArgparser

from niftymic.definitions import REGEX_FILENAMES, DIR_TMP


def main():

    time_start = ph.start_timing()

    np.set_printoptions(precision=3)

    input_parser = InputArgparser(
        description="Register an obtained reconstruction (moving) "
        "to a template image/space (fixed) using rigid registration. "
        "The resulting registration can optionally be applied to previously "
        "obtained motion correction slice transforms so that a volumetric "
        "reconstruction is possible in the (standard anatomical) space "
        "defined by the fixed.",
    )
    input_parser.add_fixed(required=True)
    input_parser.add_moving(required=True)
    input_parser.add_output(
        help="Path to registration transform (.txt)",
        required=True)
    input_parser.add_fixed_mask(required=False)
    input_parser.add_moving_mask(required=False)
    input_parser.add_option(
        option_string="--initial-transform",
        type=str,
        help="Path to initial transform. "
        "If not provided, registration will be initialized based on "
        "rigid alignment of eigenbasis of the fixed/moving image masks "
        "using principal component analysis",
        default=None)
    input_parser.add_v2v_method(
        option_string="--method",
        help="Registration method used for the registration.",
        default="RegAladin",
    )
    input_parser.add_dir_input_mc()
    input_parser.add_verbose(default=0)
    input_parser.add_log_config(default=1)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    debug = 0

    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))

    if not args.output.endswith(".txt"):
        raise IOError("output transformation path must end in '.txt'")

    dir_output = os.path.dirname(args.output)
    ph.create_directory(dir_output)

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")
    fixed = st.Stack.from_filename(
        file_path=args.fixed,
        file_path_mask=args.fixed_mask,
        extract_slices=False)
    moving = st.Stack.from_filename(
        file_path=args.moving,
        file_path_mask=args.moving_mask,
        extract_slices=False)

    path_to_tmp_output = os.path.join(
        DIR_TMP,
        ph.append_to_filename(os.path.basename(args.moving), "_warped"))

    # ---------------------------- Initialization ----------------------------
    if args.initial_transform is None:
        ph.print_title("Estimate initial transform using PCA")

        if args.moving_mask is None or args.fixed_mask is None:
            ph.print_warning("Fixed and moving masks are strongly encouraged")
        transform_initializer = tinit.TransformInitializer(
            fixed=fixed, moving=moving, similarity_measure="NMI")
        transform_initializer.run()
        transform_init_sitk = transform_initializer.get_transform_sitk()
    else:
        transform_init_sitk = sitkh.read_transform_sitk(args.initial_transform)
    sitk.WriteTransform(transform_init_sitk, args.output)

    # -------------------Register Reconstruction to Template-------------------
    ph.print_title("Registration")

    if args.method == "RegAladin":
        path_to_transform_regaladin = os.path.join(
            DIR_TMP, "transform_regaladin.txt")

        # Convert SimpleITK to RegAladin transform
        cmd = "simplereg_transform -sitk2nreg %s %s" % (
            args.output, path_to_transform_regaladin)
        ph.execute_command(cmd, verbose=False)

        # Run NiftyReg
        cmd_args = ["reg_aladin"]
        cmd_args.append("-ref %s" % args.fixed)
        cmd_args.append("-flo %s" % args.moving)
        cmd_args.append("-res %s" % path_to_tmp_output)
        cmd_args.append("-inaff %s" % path_to_transform_regaladin)
        cmd_args.append("-aff %s" % path_to_transform_regaladin)
        cmd_args.append("-rigOnly")
        cmd_args.append("-voff")
        if args.fixed_mask is not None:
            cmd_args.append("-rmask %s" % args.fixed_mask)

        # To avoid error "0 correspondances between blocks were found" that can
        # occur for some cases. Also, disable moving mask, as this would be ignored
        # anyway
        cmd_args.append("-noSym")
        # if args.moving_mask is not None:
        #     cmd_args.append("-fmask %s" % args.moving_mask)

        ph.print_info("Run Registration (RegAladin) ... ", newline=False)
        ph.execute_command(" ".join(cmd_args), verbose=False)
        print("done")

        # Convert RegAladin to SimpleITK transform
        cmd = "simplereg_transform -nreg2sitk %s %s" % (
            path_to_transform_regaladin, args.output)
        ph.execute_command(cmd, verbose=False)

    else:
        path_to_transform_flirt = os.path.join(DIR_TMP, "transform_flirt.txt")

        # Convert SimpleITK into FLIRT transform
        cmd = "simplereg_transform -sitk2flirt %s %s %s %s" % (
            args.output, args.fixed, args.moving, path_to_transform_flirt)
        ph.execute_command(cmd, verbose=False)

        # Define search angle ranges for FLIRT in all three dimensions
        search_angles = ["-searchr%s -%d %d" % (x, 180, 180)
                         for x in ["x", "y", "z"]]

        cmd_args = ["flirt"]
        cmd_args.append("-in %s" % args.moving)
        cmd_args.append("-ref %s" % args.fixed)
        if args.initial_transform is not None:
            cmd_args.append("-init %s" % path_to_transform_flirt)
        cmd_args.append("-omat %s" % path_to_transform_flirt)
        cmd_args.append("-out %s" % path_to_tmp_output)
        cmd_args.append("-dof 6")
        cmd_args.append((" ").join(search_angles))
        if args.moving_mask is not None:
            cmd_args.append("-inweight %s" % args.moving_mask)
        if args.fixed_mask is not None:
            cmd_args.append("-refweight %s" % args.fixed_mask)
        ph.print_info("Run Registration (FLIRT) ... ", newline=False)
        ph.execute_command(" ".join(cmd_args), verbose=False)
        print("done")

        # Convert FLIRT to SimpleITK transform
        cmd = "simplereg_transform -flirt2sitk %s %s %s %s" % (
            path_to_transform_flirt, args.fixed, args.moving, args.output)
        ph.execute_command(cmd, verbose=False)

    if debug:
        ph.show_niftis([args.fixed, path_to_tmp_output])

    if args.dir_input_mc is not None:
        transform_sitk = sitkh.read_transform_sitk(
            args.output, inverse=1)

        if args.dir_input_mc.endswith("/"):
            subdir_mc = args.dir_input_mc.split("/")[-2]
        else:
            subdir_mc = args.dir_input_mc.split("/")[-1]
        dir_output_mc = os.path.join(dir_output, subdir_mc)

        ph.create_directory(dir_output_mc, delete_files=True)
        pattern = REGEX_FILENAMES + "[.]tfm"
        p = re.compile(pattern)
        trafos = [t for t in os.listdir(args.dir_input_mc) if p.match(t)]
        for t in trafos:
            path_to_input_transform = os.path.join(args.dir_input_mc, t)
            path_to_output_transform = os.path.join(dir_output_mc, t)
            t_sitk = sitkh.read_transform_sitk(path_to_input_transform)
            t_sitk = sitkh.get_composite_sitk_affine_transform(
                transform_sitk, t_sitk)
            sitk.WriteTransform(t_sitk, path_to_output_transform)

    if args.verbose:
        ph.show_niftis([args.fixed, path_to_tmp_output])

    elapsed_time_total = ph.stop_timing(time_start)

    # Summary
    ph.print_title("Summary")
    print("Computational Time: %s" % (elapsed_time_total))

    return 0


if __name__ == '__main__':
    main()
