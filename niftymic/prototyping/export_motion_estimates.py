##
# \file export_motion_estimates.py
# \brief      Script to export stack motion estimates
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2018
#

import os
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh
import simplereg.utilities

import niftymic.base.stack as st
import niftymic.base.data_reader as dr
import niftymic.base.data_writer as dw
import niftymic.utilities.motion_updater as mu
from niftymic.utilities.input_arparser import InputArgparser


def main():

    time_start = ph.start_timing()

    # Set print options for numpy
    np.set_printoptions(precision=3)

    # Read input
    input_parser = InputArgparser(
        description="Export image that illustrates the estimated "
        "intra-stack motion by showing the individual voxel displacements for "
        "all slices in millimetre.",
    )

    input_parser.add_filenames(required=True)
    input_parser.add_filenames_masks()
    input_parser.add_dir_input_mc(required=True)
    input_parser.add_dir_output(required=True)
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_log_config(default=False)
    input_parser.add_verbose(default=0)
    input_parser.add_argument(
        "--multi-component", "-multi-component",
        action='store_true',
        help="If given, the input image is interpreted as multi-component."
    )
    input_parser.add_option(
        option_string="--suffix-output",
        help="Suffix that is being added to each input filename to define "
        "output filename at dir-output.",
        type=str,
        default="_disp_s2v",
    )

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))
    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")

    if args.multi_component:
        if len(args.filenames) > 1:
            raise IOError(
                "Only one multi-component input image can be processed")

        if args.filenames_masks is None:
            filename_mask = None
        else:
            filename_mask = args.filenames_masks[0]
        ph.print_info("Read multi-component image ...")
        data_reader = dr.MultiComponentImageReader(
            path_to_image=args.filenames[0],
            path_to_image_mask=filename_mask,
        )

    else:
        data_reader = dr.MultipleImagesReader(
            file_paths=args.filenames,
            file_paths_masks=args.filenames_masks,
            suffix_mask=args.suffix_mask,
        )

    data_reader.read_data()
    stacks_orig = data_reader.get_data()
    ph.print_info("%d input stacks read for further processing" %
                  len(stacks_orig))

    # ------------------------------Update Motion------------------------------
    motion_updater = mu.MotionUpdater(
        stacks=stacks_orig,
        dir_motion_correction=args.dir_input_mc,
    )
    motion_updater.run()
    stacks = motion_updater.get_data()
    stacks_motion = [None] * len(stacks)

    # ---------------Build image(s) to visualize applied motion---------------
    ph.print_title("Visualize Motion")
    ph.print_info("Compute voxel displacements ...")
    for i, stack in enumerate(stacks):
        shape = stack.sitk.GetSize()[::-1]

        # Compute displacements for individual slice corrections
        nda = np.ones(shape) * np.inf
        for slice in stack.get_slices():
            j = slice.get_slice_number()
            slice_orig = stacks_orig[i].get_slice(j)

            # Get slice-to-volume transformation
            transform_sitk = slice.get_motion_correction_transform()

            # Get volume-to-volume transformation
            # transform_sitk = slice.get_registration_history()[1][1]

            # Compute voxel displacements w.r.t. original slice position
            nda[j, ...] = simplereg.utilities.get_voxel_displacements(
                slice_orig.sitk, transform_sitk)

        # Create sitk.Image object
        stack_sitk = sitk.GetImageFromArray(nda)
        stack_sitk.SetDirection(stacks_orig[i].sitk.GetDirection())
        stack_sitk.SetSpacing(stacks_orig[i].sitk.GetSpacing())
        stack_sitk.SetOrigin(stacks_orig[i].sitk.GetOrigin())

        # Create stack object
        stacks_motion[i] = st.Stack.from_sitk_image(
            image_sitk=stack_sitk,
            filename="%s%s" % (
                stacks_orig[i].get_filename(), args.suffix_output),
            slice_thickness=stacks_orig[i].get_slice_thickness(),
            extract_slices=False,
        )

    # -------------------------------Write Data--------------------------------
    paths_to_output = []
    if args.multi_component:
        path_to_output = os.path.join(
            args.dir_output,
            ph.append_to_filename(os.path.basename(args.filenames[0]),
                                  args.suffix_output)
        )
        data_writer = dw.MultiComponentImageWriter(
            stacks_motion, path_to_output)
        data_writer.write_data()

        # verbose
        paths_to_output.append(path_to_output)

    else:
        data_writer = dw.MultipleStacksWriter(stacks_motion, args.dir_output)
        data_writer.write_data()

        # verbose
        for s in stacks_motion:
            paths_to_output.append(
                os.path.join(args.dir_output, "%s.nii.gz" % filename))

    if args.verbose:
        ph.show_niftis(paths_to_output)

    ph.print_title("Summary")
    print("Computational Time: %s" % ph.stop_timing(time_start))

    return 0


if __name__ == '__main__':
    main()
