##
# \file register_image.py
# \brief      Script to register the obtained reconstruction to a template
#             space.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       October 2017
#

import numpy as np
import os

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.data_reader as dr
import niftymic.base.stack as st
import niftymic.registration.flirt as regflirt
import niftymic.registration.niftyreg as niftyreg
from niftymic.utilities.input_arparser import InputArgparser


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
    input_parser.add_moving(
        required=True,
        nargs="+",
        help="Specify moving image to be warped to fixed space. "
        "If multiple images are provided, all images will be transformed "
        "uniformly according to the registration obtained for the first one."
    )
    input_parser.add_dir_output(required=True)
    input_parser.add_dir_input()
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_search_angle(default=180)
    input_parser.add_option(
        option_string="--transform-only",
        type=int,
        help="Turn on/off functionality to transform moving image(s) to fixed "
        "image only, i.e. no resampling to fixed image space",
        default=0)
    input_parser.add_option(
        option_string="--write-transform",
        type=int,
        help="Turn on/off functionality to write registration transform",
        default=0)
    input_parser.add_option(
        option_string="--use-fixed-mask",
        type=int,
        help="Turn on/off functionality to use fixed image mask during "
        "registration.",
        default=0)
    input_parser.add_option(
        option_string="--use-moving-mask",
        type=int,
        help="Turn on/off functionality to use moving image mask during "
        "registration.",
        default=0)
    input_parser.add_verbose(default=0)
    input_parser.add_log_script_execution(default=1)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    use_reg_aladin_for_refinement = True

    # Write script execution call
    if args.log_script_execution:
        input_parser.write_performed_script_execution(
            os.path.abspath(__file__))

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")
    data_reader = dr.MultipleImagesReader(args.moving, suffix_mask="_mask")
    data_reader.read_data()
    moving = data_reader.get_data()

    data_reader = dr.MultipleImagesReader([args.fixed], suffix_mask="_mask")
    data_reader.read_data()
    fixed = data_reader.get_data()[0]

    # -------------------Register Reconstruction to Template-------------------
    ph.print_title("Register Reconstruction to Template")

    # Define search angle ranges for FLIRT in all three dimensions
    search_angles = ["-searchr%s -%d %d" %
                     (x, args.search_angle, args.search_angle)
                     for x in ["x", "y", "z"]]
    search_angles = (" ").join(search_angles)
    options_args = []
    options_args.append(search_angles)
    # cost = "mutualinfo"
    # options_args.append("-searchcost %s -cost %s" % (cost, cost))
    registration = regflirt.FLIRT(
        fixed=moving[0],
        moving=fixed,
        use_fixed_mask=args.use_fixed_mask,
        use_moving_mask=args.use_moving_mask,
        registration_type="Rigid",
        use_verbose=False,
        options=(" ").join(options_args),
    )
    ph.print_info("Run Registration (FLIRT) ... ", newline=False)
    registration.run()
    print("done")
    transform_sitk = registration.get_registration_transform_sitk()

    if args.write_transform:
        path_to_transform = os.path.join(
            args.dir_output, "registration_transform_sitk.txt")
        sitk.WriteTransform(transform_sitk, path_to_transform)

    # Apply rigidly transform to align reconstruction (moving) with template
    # (fixed)
    for m in moving:
        m.update_motion_correction(transform_sitk)

        # Additionally, use RegAladin for more accurate alignment
        # Rationale: FLIRT has better capture range, but RegAladin seems to
        # find better alignment once it is within its capture range.
        if use_reg_aladin_for_refinement:
            registration = niftyreg.RegAladin(
                fixed=m,
                use_fixed_mask=args.use_fixed_mask,
                use_moving_mask=args.use_moving_mask,
                moving=fixed,
                registration_type="Rigid",
                use_verbose=False,
            )
            ph.print_info("Run Registration (RegAladin) ... ", newline=False)
            registration.run()
            print("done")
            transform2_sitk = registration.get_registration_transform_sitk()
            m.update_motion_correction(transform2_sitk)
            transform_sitk = sitkh.get_composite_sitk_affine_transform(
                transform2_sitk, transform_sitk)

    if args.transform_only:
        for m in moving:
            m.write(args.dir_output, write_mask=False)
        ph.exit()

    # Resample reconstruction (moving) to template space (fixed)
    warped_moving = [m.get_resampled_stack(fixed.sitk, interpolator="Linear")
                     for m in moving]

    for wm in warped_moving:
        wm.set_filename(
            wm.get_filename() + "ResamplingToTemplateSpace")

        if args.verbose:
            sitkh.show_stacks([fixed, wm], segmentation=fixed)

        # Write resampled reconstruction (moving)
        wm.write(args.dir_output, write_mask=False)

    if args.dir_input is not None:
        data_reader = dr.ImageSlicesDirectoryReader(
            path_to_directory=args.dir_input,
            suffix_mask=args.suffix_mask)
        data_reader.read_data()
        stacks = data_reader.get_data()

        for i, stack in enumerate(stacks):
            stack.update_motion_correction(transform_sitk)
            ph.print_info("Stack %d/%d: All slice transforms updated" %
                          (i + 1, len(stacks)))

            # Write transformed slices
            stack.write(
                os.path.join(args.dir_output, "motion_correction"),
                write_mask=True,
                write_slices=True,
                write_transforms=True,
                suffix_mask=args.suffix_mask,
            )

    elapsed_time_total = ph.stop_timing(time_start)

    # Summary
    ph.print_title("Summary")
    print("Computational Time: %s" % (elapsed_time_total))

    return 0

if __name__ == '__main__':
    main()
