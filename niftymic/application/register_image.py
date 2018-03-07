##
# \file register_image.py
# \brief      Script to register the obtained reconstruction to a template
#             space.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       October 2017
#

import os
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.data_reader as dr
import niftymic.base.stack as st
import niftymic.registration.flirt as regflirt
import niftymic.registration.niftyreg as niftyreg
import niftymic.validation.image_similarity_evaluator as ise
from niftymic.utilities.input_arparser import InputArgparser


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
def get_ap_flip_transform(stack, initializer_type="GEOMETRY"):
    initial_transform = sitk.CenteredTransformInitializer(
        stack.sitk,
        stack.sitk,
        sitk.Euler3DTransform(),
        eval("sitk.CenteredTransformInitializerFilter.%s" % (
            initializer_type)))

    translation = np.array(initial_transform.GetFixedParameters()[0:3])

    transform_sitk1 = sitk.Euler3DTransform()
    transform_sitk1.SetTranslation(-translation)

    transform_sitk2 = sitk.Euler3DTransform()
    transform_sitk2.SetRotation(0, 0, np.pi)

    transform_sitk3 = sitk.Euler3DTransform(transform_sitk1.GetInverse())

    transform_sitk = sitkh.get_composite_sitk_euler_transform(
        transform_sitk2, transform_sitk1)
    transform_sitk = sitkh.get_composite_sitk_euler_transform(
        transform_sitk3, transform_sitk)

    # sitkh.show_sitk_image((
    #     [stack.sitk,
    #     sitkh.get_transformed_sitk_image(stack.sitk, transform_sitk)]))

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
        option_string="--initial-transform",
        type=str,
        help="Set initial transform to be used.",
        default=None)
    input_parser.add_option(
        option_string="--write-transform",
        type=int,
        help="Turn on/off functionality to write registration transform",
        default=1)
    input_parser.add_option(
        option_string="--use-fixed-mask",
        type=int,
        help="Turn on/off functionality to use fixed image mask during "
        "registration. It is defined via 'mask-suffix' and must be in the "
        "same directory as the fixed image.",
        default=0)
    input_parser.add_option(
        option_string="--use-moving-mask",
        type=int,
        help="Turn on/off functionality to use moving image mask during "
        "registration. It is defined via 'mask-suffix' and must be in the "
        "same directory as the moving image(s).",
        default=0)
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
    input_parser.add_log_script_execution(default=1)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    debug = 0

    # Write script execution call
    if args.log_script_execution:
        input_parser.write_performed_script_execution(
            os.path.abspath(__file__))

    if not args.use_regaladin and not args.use_flirt:
        raise IOError("Either RegAladin or FLIRT must be activated.")

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")
    data_reader = dr.MultipleImagesReader(args.moving, suffix_mask="_mask")
    data_reader.read_data()
    moving = data_reader.get_data()

    data_reader = dr.MultipleImagesReader([args.fixed], suffix_mask="_mask")
    data_reader.read_data()
    fixed = data_reader.get_data()[0]

    if args.initial_transform is not None:
        data_reader = dr.MultipleTransformationsReader(
            [args.initial_transform])
        data_reader.read_data()
        transform_sitk = data_reader.get_data()[0]
        moving[0].update_motion_correction(transform_sitk)
    else:
        transform_sitk = sitk.AffineTransform(fixed.sitk.GetDimension())

    # -------------------Register Reconstruction to Template-------------------
    ph.print_title("Register Reconstruction to Template")

    if args.use_flirt:
        # Define search angle ranges for FLIRT in all three dimensions
        search_angles = ["-searchr%s -%d %d" %
                         (x, args.search_angle, args.search_angle)
                         for x in ["x", "y", "z"]]
        search_angles = (" ").join(search_angles)
        options_args = []
        options_args.append(search_angles)
        # cost = "mutualinfo"
        # options_args.append("-cost %s" % (cost))
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
        transform2_sitk = registration.get_registration_transform_sitk()
        moving[0].update_motion_correction(transform2_sitk)

        transform_sitk = sitkh.get_composite_sitk_affine_transform(
            transform2_sitk, transform_sitk)

        if debug:
            sitkh.show_stacks([fixed, moving[0]], segmentation=fixed)

    # Additionally, use RegAladin for more accurate alignment
    # Rationale: FLIRT has better capture range, but RegAladin seems to
    # find better alignment once it is within its capture range.
    if args.use_regaladin:
        registration = niftyreg.RegAladin(
            fixed=moving[0],
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
        moving[0].update_motion_correction(transform2_sitk)
        transform_sitk = sitkh.get_composite_sitk_affine_transform(
            transform2_sitk, transform_sitk)

        if debug:
            sitkh.show_stacks([fixed, moving[0]], segmentation=fixed)

    if args.test_ap_flip:
        moving0_flipped = st.Stack.from_stack(moving[0])
        moving0_flipped.set_filename("%s_flipped" % moving[0].get_filename())

        transform_ap_flip_sitk = get_ap_flip_transform(moving0_flipped)

        moving0_flipped.update_motion_correction(transform_ap_flip_sitk)
        registration = niftyreg.RegAladin(
            fixed=moving0_flipped,
            use_fixed_mask=args.use_fixed_mask,
            use_moving_mask=args.use_moving_mask,
            moving=fixed,
            registration_type="Rigid",
            use_verbose=False,
        )
        ph.print_info("Run Registration AP-flipped (RegAladin) ... ",
                      newline=False)
        registration.run()
        print("done")

        transform2_sitk = registration.get_registration_transform_sitk()
        moving0_flipped.update_motion_correction(transform2_sitk)

        stacks = [s.get_resampled_stack(fixed.sitk)
                  for s in [moving[0], moving0_flipped]]
        image_similarity_evaluator = ise.ImageSimilarityEvaluator(
            stacks=stacks, reference=fixed)
        image_similarity_evaluator.compute_similarities()
        similarities = image_similarity_evaluator.get_similarities()

        if similarities["NMI"][1] > similarities["NMI"][0]:
            ph.print_info("AP-flipped outcome better")
            transform_update_sitk = sitkh.get_composite_sitk_affine_transform(
                transform2_sitk, transform_ap_flip_sitk)
            moving[0].update_motion_correction(transform_update_sitk)

            transform_sitk = sitkh.get_composite_sitk_affine_transform(
                transform_update_sitk, transform_sitk)
        else:
            ph.print_info("AP-flip does not improve outcome")

    if args.write_transform:
        path_to_transform = os.path.join(
            args.dir_output, "registration_transform_sitk.txt")
        sitk.WriteTransform(transform_sitk, path_to_transform)
        ph.print_info("Registration transform written to '%s'" %
                      path_to_transform)

    # Apply rigidly transform to align reconstruction (moving) with template
    # (fixed)
    for m in moving[1:]:
        m.update_motion_correction(transform_sitk)

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
