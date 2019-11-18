##
# \file propagate_mask.py
# \brief      Script to propagate an image mask using rigid registration
#
# \author     Michael Ebner (michael.ebner@kcl.ac.uk)
# \date       Aug 2019
#

import os
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.data_writer as dw
import niftymic.base.stack as st
import niftymic.registration.flirt as regflirt
import niftymic.registration.niftyreg as niftyreg
import niftymic.utilities.stack_mask_morphological_operations as stmorph
from niftymic.utilities.input_arparser import InputArgparser

from niftymic.definitions import V2V_METHOD_OPTIONS, ALLOWED_EXTENSIONS


def main():

    time_start = ph.start_timing()

    # Set print options for numpy
    np.set_printoptions(precision=3)

    input_parser = InputArgparser(
        description="Propagate image mask using rigid registration.",
    )
    input_parser.add_moving(required=True)
    input_parser.add_moving_mask(required=True)
    input_parser.add_fixed(required=True)
    input_parser.add_output(required=True)
    input_parser.add_v2v_method(
        option_string="--method",
        help="Registration method used for the registration (%s)." % (
            ", or ".join(V2V_METHOD_OPTIONS)),
        default="RegAladin",
    )
    input_parser.add_option(
        option_string="--use-moving-mask",
        type=int,
        help="Turn on/off use of moving mask to constrain the registration.",
        default=0,
    )
    input_parser.add_dilation_radius(default=1)
    input_parser.add_verbose(default=0)
    input_parser.add_log_config(default=0)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    if np.alltrue([not args.output.endswith(t) for t in ALLOWED_EXTENSIONS]):
        raise ValueError(
            "output filename invalid; allowed extensions are: %s" %
            ", ".join(ALLOWED_EXTENSIONS))

    if args.method not in V2V_METHOD_OPTIONS:
        raise ValueError("method must be in {%s}" % (
            ", ".join(V2V_METHOD_OPTIONS)))

    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))

    stack = st.Stack.from_filename(
        file_path=args.fixed,
        extract_slices=False,
    )
    template = st.Stack.from_filename(
        file_path=args.moving,
        file_path_mask=args.moving_mask,
        extract_slices=False,
    )

    if args.method == "FLIRT":
        # Define search angle ranges for FLIRT in all three dimensions
        # search_angles = ["-searchr%s -%d %d" %
        #                  (x, args.search_angle, args.search_angle)
        #                  for x in ["x", "y", "z"]]
        # options = (" ").join(search_angles)
        # options += " -noresample"

        registration = regflirt.FLIRT(
            registration_type="Rigid",
            fixed=stack,
            moving=template,
            use_fixed_mask=False,
            use_moving_mask=args.use_moving_mask,
            # options=options,
            use_verbose=False,
        )
    else:
        registration = niftyreg.RegAladin(
            registration_type="Rigid",
            fixed=stack,
            moving=template,
            use_fixed_mask=False,
            use_moving_mask=args.use_moving_mask,
            # options="-ln 2",
            use_verbose=False,
        )

    try:
        registration.run()
    except RuntimeError as e:
        raise RuntimeError(
            "%s\n\n"
            "Have you tried running the script with '--use-moving-mask 0'?" % e)

    transform_sitk = registration.get_registration_transform_sitk()
    stack.sitk_mask = sitk.Resample(
        template.sitk_mask,
        stack.sitk_mask,
        transform_sitk,
        sitk.sitkNearestNeighbor,
        0,
        template.sitk_mask.GetPixelIDValue()
    )
    if args.dilation_radius > 0:
        stack_mask_morpher = stmorph.StackMaskMorphologicalOperations.from_sitk_mask(
            mask_sitk=stack.sitk_mask,
            dilation_radius=args.dilation_radius,
            dilation_kernel="Ball",
            use_dilation_in_plane_only=True,
        )
        stack_mask_morpher.run_dilation()
        stack.sitk_mask = stack_mask_morpher.get_processed_mask_sitk()

    dw.DataWriter.write_mask(stack.sitk_mask, args.output)

    elapsed_time = ph.stop_timing(time_start)

    if args.verbose:
        ph.show_nifti(args.fixed, segmentation=args.output)

    ph.print_title("Summary")
    exe_file_info = os.path.basename(os.path.abspath(__file__)).split(".")[0]
    print("%s | Computational Time for Segmentation Propagation: %s" % (
        exe_file_info, elapsed_time))

    return 0


if __name__ == '__main__':
    main()
