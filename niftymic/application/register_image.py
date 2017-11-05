##
# \file register_image.py
# \brief      Script to register the obtained reconstruction to a template
#             space.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       October 2017
#

# Import libraries
import numpy as np
import os

# Import modules
import niftymic.base.data_reader as dr
import niftymic.base.stack as st
import niftymic.registration.flirt as regflirt
import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh
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
    input_parser.add_moving(required=True)
    input_parser.add_dir_output(required=True)
    input_parser.add_dir_input()
    input_parser.add_moving_mask()
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_search_angle(default=180)
    input_parser.add_option(
        option_string="--transform-only",
        type=int,
        help="Turn on/off functionality to transform moving image to fixed "
        "image only, i.e. no resampling to fixed image space",
        default=0)
    input_parser.add_option(
        option_string="--write-transform",
        type=int,
        help="Turn on/off functionality to write registration transform",
        default=0)
    input_parser.add_verbose(default=0)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")
    moving = st.Stack.from_filename(args.moving, args.moving_mask)

    data_reader = dr.MultipleImagesReader([args.fixed], suffix_mask="_mask")
    data_reader.read_data()
    fixed = data_reader.get_stacks()[0]

    # -------------------Register Reconstruction to Template-------------------
    ph.print_title("Register Reconstruction to Template")

    # Define search angle ranges for FLIRT in all three dimensions
    search_angles = ["-searchr%s -%d %d" %
                     (x, args.search_angle, args.search_angle)
                     for x in ["x", "y", "z"]]
    search_angles = (" ").join(search_angles)

    registration = regflirt.FLIRT(
        fixed=moving,
        moving=fixed,
        registration_type="Rigid",
        use_verbose=False,
        options=search_angles,
    )
    ph.print_info("Run Registration ... ", newline=False)
    registration.run()
    print("done")
    transform_sitk = registration.get_registration_transform_sitk()

    if args.write_transform:
        path_to_transform = os.path.join(
            args.dir_output, "registration_transform_sitk.txt")
        sitk.WriteTransform(transform_sitk, path_to_transform)

    # Apply rigidly transform to align reconstruction (moving) with template
    # (fixed)
    moving.update_motion_correction(transform_sitk)

    if args.transform_only:
        moving.write(args.dir_output, write_mask=True)
        ph.exit()

    # Resample reconstruction (moving) to template space (fixed)
    warped_moving = \
        moving.get_resampled_stack(fixed.sitk)
    warped_moving.set_filename(
        warped_moving.get_filename() + "ResamplingToTemplateSpace")

    # Write resampled reconstruction (moving)
    if args.moving_mask is not None:
        write_mask = True
    else:
        write_mask = False
    warped_moving.write(args.dir_output, write_mask=write_mask)

    if args.dir_input is not None:
        data_reader = dr.ImageSlicesDirectoryReader(
            path_to_directory=args.dir_input,
            suffix_mask=args.suffix_mask)
        data_reader.read_data()
        stacks = data_reader.get_stacks()

        for i, stack in enumerate(stacks):
            stack.update_motion_correction(transform_sitk)
            ph.print_info("Stack %d/%d: All slice transforms updated" %
                          (i+1, len(stacks)))

            # Write transformed slices
            stack.write(
                os.path.join(args.dir_output, "motion_correction"),
                write_mask=True,
                write_slices=True,
                write_transforms=True,
                suffix_mask=args.suffix_mask,
            )

    if args.verbose:
        tmp = warped_moving.get_stack_multiplied_with_mask()
        tmp.set_filename(moving.get_filename() + "_times_mask")
        sitkh.show_stacks([fixed, warped_moving, tmp],
                          segmentation=warped_moving)

    elapsed_time_total = ph.stop_timing(time_start)

    # Summary
    ph.print_title("Summary")
    print("Computational Time: %s" % (elapsed_time_total))

    return 0

if __name__ == '__main__':
    main()
