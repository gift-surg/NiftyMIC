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
# import nipype.interfaces.fsl
# import nipype.interfaces.niftyreg

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.data_reader as dr
import niftymic.base.stack as st
import niftymic.registration.flirt as regflirt
import niftymic.registration.niftyreg as niftyreg
import niftymic.validation.image_similarity_evaluator as ise
from niftymic.utilities.input_arparser import InputArgparser

from niftymic.definitions import REGEX_FILENAMES, DIR_TMP


##
# Gets the ap flip transform in case the 'stack' physical coordinate is aligned
# with the voxel space
# \date       2018-02-07 19:59:39+0000
#
# \param      stack             The stack
# \param      initializer_type  The initializer type
#
# \return     The ap flip transform.
#
def get_flip_transform(path_to_image, initializer_type="GEOMETRY", flip="AP"):
    image_sitk = sitk.ReadImage(path_to_image)
    initial_transform = sitk.CenteredTransformInitializer(
        image_sitk,
        image_sitk,
        sitk.Euler3DTransform(),
        eval("sitk.CenteredTransformInitializerFilter.%s" % (
            initializer_type)))

    translation = np.array(initial_transform.GetFixedParameters()[0:3])

    transform_sitk1 = sitk.Euler3DTransform()
    transform_sitk1.SetTranslation(-translation)

    transform_sitk2 = sitk.Euler3DTransform()

    # AP 'flip', i.e. associated rotation
    if flip == "AP":
        transform_sitk2.SetRotation(0, 0, np.pi)

    # SI 'flip', i.e. associated rotation
    elif flip == "SI":
        transform_sitk2.SetRotation(0, np.pi, 0)

    # SI 'flip', i.e. associated rotation
    elif flip == "RL":
        transform_sitk2.SetRotation(np.pi, 0, 0)

    # AP+SI 'flip', i.e. associated rotation
    else:
        transform_sitk2.SetRotation(0, np.pi, np.pi)

    transform_sitk3 = sitk.Euler3DTransform(transform_sitk1.GetInverse())

    transform_sitk = sitkh.get_composite_sitk_euler_transform(
        transform_sitk2, transform_sitk1)
    transform_sitk = sitkh.get_composite_sitk_euler_transform(
        transform_sitk3, transform_sitk)

    # image = sitk.ReadImage(path_to_image)
    # sitkh.show_sitk_image((
    #     [image, sitkh.get_transformed_sitk_image(image, transform_sitk)]),
    #     label=["orig", "%s-flipped" % flip]
    # )

    return transform_sitk


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
    input_parser.add_fixed_mask()
    input_parser.add_moving_mask()
    input_parser.add_dir_input_mc()
    input_parser.add_search_angle(default=180)
    input_parser.add_option(
        option_string="--initial-transform",
        type=str,
        help="Path to initial transform.",
        default=None)
    input_parser.add_option(
        option_string="--test-ap-flip",
        type=int,
        help="Turn on/off functionality to run an additional registration "
        "after an AP-flip. Seems to be more robust to find a better "
        "registration outcome in general.",
        default=1)
    input_parser.add_option(
        option_string="--use-flirt",
        type=int,
        help="Turn on/off functionality to use FLIRT for the registration.",
        default=1)
    input_parser.add_option(
        option_string="--use-regaladin",
        type=int,
        help="Turn on/off functionality to use RegAladin for the "
        "registration.",
        default=1)
    input_parser.add_verbose(default=0)
    input_parser.add_log_config(default=1)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    debug = 0

    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))

    if not args.use_regaladin and not args.use_flirt:
        raise IOError("Either RegAladin or FLIRT must be activated.")

    if not args.output.endswith(".txt"):
        raise IOError("output transformation path must end in '.txt'")

    dir_output = os.path.dirname(args.output)

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

    if args.initial_transform is not None:
        transform_sitk = sitkh.read_transform_sitk(args.initial_transform)
    else:
        transform_sitk = sitk.AffineTransform(fixed.sitk.GetDimension())
    sitk.WriteTransform(transform_sitk, args.output)

    path_to_tmp_output = os.path.join(
        DIR_TMP,
        ph.append_to_filename(os.path.basename(args.moving), "_warped"))

    # -------------------Register Reconstruction to Template-------------------
    ph.print_title("Register Reconstruction to Template")

    if args.use_flirt:
        path_to_transform_flirt = os.path.join(DIR_TMP, "transform_flirt.txt")

        # Convert SimpleITK into FLIRT transform
        cmd = "simplereg_transform -sitk2flirt %s %s %s %s" % (
            args.output, args.fixed, args.moving, path_to_transform_flirt)
        ph.execute_command(cmd, verbose=False)

        # Define search angle ranges for FLIRT in all three dimensions
        search_angles = ["-searchr%s -%d %d" %
                         (x, args.search_angle, args.search_angle)
                         for x in ["x", "y", "z"]]

        # flt = nipype.interfaces.fsl.FLIRT()
        # flt.inputs.in_file = args.moving
        # flt.inputs.reference = args.fixed
        # if args.initial_transform is not None:
        #     flt.inputs.in_matrix_file = path_to_transform_flirt
        # flt.inputs.out_matrix_file = path_to_transform_flirt
        # # flt.inputs.output_type = "NIFTI_GZ"
        # flt.inputs.out_file = path_to_tmp_output
        # flt.inputs.args = "-dof 6"
        # flt.inputs.args += " %s" % " ".join(search_angles)
        # if args.moving_mask is not None:
        #     flt.inputs.in_weight = args.moving_mask
        # if args.fixed_mask is not None:
        #     flt.inputs.ref_weight = args.fixed_mask
        # ph.print_info("Run Registration (FLIRT) ... ", newline=False)
        # flt.run()
        # print("done")

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

    # Additionally, use RegAladin for more accurate alignment
    # Rationale: FLIRT has better capture range, but RegAladin seems to
    # find better alignment once it is within its capture range.
    if args.use_regaladin:
        path_to_transform_regaladin = os.path.join(
            DIR_TMP, "transform_regaladin.txt")

        # Convert SimpleITK to RegAladin transform
        cmd = "simplereg_transform -sitk2nreg %s %s" % (
            args.output, path_to_transform_regaladin)
        ph.execute_command(cmd, verbose=False)

        # nreg = nipype.interfaces.niftyreg.RegAladin()
        # nreg.inputs.ref_file = args.fixed
        # nreg.inputs.flo_file = args.moving
        # nreg.inputs.res_file = path_to_tmp_output
        # nreg.inputs.in_aff_file = path_to_transform_regaladin
        # nreg.inputs.aff_file = path_to_transform_regaladin
        # nreg.inputs.args = "-rigOnly -voff"
        # if args.moving_mask is not None:
        #     nreg.inputs.fmask_file = args.moving_mask
        # if args.fixed_mask is not None:
        #     nreg.inputs.rmask_file = args.fixed_mask
        # ph.print_info("Run Registration (RegAladin) ... ", newline=False)
        # nreg.run()
        # print("done")

        cmd_args = ["reg_aladin"]
        cmd_args.append("-ref %s" % args.fixed)
        cmd_args.append("-flo %s" % args.moving)
        cmd_args.append("-res %s" % path_to_tmp_output)
        if args.initial_transform is not None or args.use_flirt == 1:
            cmd_args.append("-inaff %s" % path_to_transform_regaladin)
        cmd_args.append("-aff %s" % path_to_transform_regaladin)
        # cmd_args.append("-cog")
        # cmd_args.append("-ln 2")
        cmd_args.append("-rigOnly")
        cmd_args.append("-voff")
        if args.moving_mask is not None:
            cmd_args.append("-fmask %s" % args.moving_mask)
        if args.fixed_mask is not None:
            cmd_args.append("-rmask %s" % args.fixed_mask)
        ph.print_info("Run Registration (RegAladin) ... ", newline=False)
        ph.execute_command(" ".join(cmd_args), verbose=False)
        print("done")

        # Convert RegAladin to SimpleITK transform
        cmd = "simplereg_transform -nreg2sitk %s %s" % (
            path_to_transform_regaladin, args.output)
        ph.execute_command(cmd, verbose=False)

        if debug:
            ph.show_niftis([args.fixed, path_to_tmp_output])

    if args.test_ap_flip:
        path_to_transform_flip = os.path.join(DIR_TMP, "transform_flip.txt")
        path_to_tmp_output_flip = os.path.join(DIR_TMP, "output_flip.nii.gz")

        for flip in [
            "AP",
            "SI",
            # "RL",
        ]:

            # Get AP-flip transform
            transform_flip_sitk = get_flip_transform(args.fixed, flip=flip)
            path_to_transform_flip_regaladin = os.path.join(
                DIR_TMP, "transform_flip_regaladin.txt")
            sitk.WriteTransform(transform_flip_sitk, path_to_transform_flip)

            # Compose current transform with AP flip transform
            cmd = "simplereg_transform -c %s %s %s" % (
                args.output, path_to_transform_flip, path_to_transform_flip)
            ph.execute_command(cmd, verbose=False)

            # Convert SimpleITK to RegAladin transform
            cmd = "simplereg_transform -sitk2nreg %s %s" % (
                path_to_transform_flip, path_to_transform_flip_regaladin)
            ph.execute_command(cmd, verbose=False)

            # nreg = nipype.interfaces.niftyreg.RegAladin()
            # nreg.inputs.ref_file = args.fixed
            # nreg.inputs.flo_file = args.moving
            # nreg.inputs.res_file = path_to_tmp_output_flip
            # nreg.inputs.in_aff_file = path_to_transform_flip_regaladin
            # nreg.inputs.aff_file = path_to_transform_flip_regaladin
            # nreg.inputs.args = "-rigOnly -voff"
            # if args.moving_mask is not None:
            #     nreg.inputs.fmask_file = args.moving_mask
            # if args.fixed_mask is not None:
            #     nreg.inputs.rmask_file = args.fixed_mask
            # ph.print_info("Run Registration AP-flipped (RegAladin) ... ",
            #               newline=False)
            # nreg.run()
            # print("done")

            cmd_args = ["reg_aladin"]
            cmd_args.append("-ref %s" % args.fixed)
            cmd_args.append("-flo %s" % args.moving)
            cmd_args.append("-res %s" % path_to_tmp_output_flip)
            cmd_args.append("-inaff %s" % path_to_transform_flip_regaladin)
            cmd_args.append("-aff %s" % path_to_transform_flip_regaladin)
            cmd_args.append("-rigOnly")
            # cmd_args.append("-ln 2")
            cmd_args.append("-voff")
            if args.moving_mask is not None:
                cmd_args.append("-fmask %s" % args.moving_mask)
            if args.fixed_mask is not None:
                cmd_args.append("-rmask %s" % args.fixed_mask)
            ph.print_info("Run Registration %s-flipped (RegAladin) ... " %
                          flip, newline=False)
            ph.execute_command(" ".join(cmd_args), verbose=False)
            print("done")

            if debug:
                ph.show_niftis(
                    [args.fixed, path_to_tmp_output, path_to_tmp_output_flip])

            warped_moving = st.Stack.from_filename(
                path_to_tmp_output, extract_slices=False)
            warped_moving_flip = st.Stack.from_filename(
                path_to_tmp_output_flip, extract_slices=False)
            fixed = st.Stack.from_filename(args.fixed, args.fixed_mask)

            stacks = [warped_moving, warped_moving_flip]
            image_similarity_evaluator = ise.ImageSimilarityEvaluator(
                stacks=stacks, reference=fixed)
            image_similarity_evaluator.compute_similarities()
            similarities = image_similarity_evaluator.get_similarities()

            if similarities["NMI"][1] > similarities["NMI"][0]:
                ph.print_info("%s-flipped outcome better" % flip)

                # Convert RegAladin to SimpleITK transform
                cmd = "simplereg_transform -nreg2sitk %s %s" % (
                    path_to_transform_flip_regaladin, args.output)
                ph.execute_command(cmd, verbose=False)

                # Copy better outcome
                cmd = "cp -p %s %s" % (path_to_tmp_output_flip,
                                       path_to_tmp_output)
                ph.execute_command(cmd, verbose=False)

            else:
                ph.print_info("%s-flip does not improve outcome" % flip)

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
