##
# \file simulate_stacks_from_reconstruction.py
# \brief      Simulate stacks from obtained reconstruction
#
# Example call:
# python simulate_stacks_from_reconstruction.py \
# --dir-input dir-to-motion-correction \
# --reconstruction volumetric_reconstruction.nii.gz \
# --copy-data 1 \
# --dir-output dir-to-output
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       November 2017
#

# Import libraries
import os
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.stack as st
import niftymic.base.data_reader as dr
import niftymic.base.data_writer as dw
import niftymic.reconstruction.linear_operators as lin_op
from niftymic.utilities.input_arparser import InputArgparser
from niftymic.definitions import ALLOWED_INTERPOLATORS

INTERPOLATOR_TYPES = "%s, or %s" % (
    (", ").join(ALLOWED_INTERPOLATORS[0:-1]), ALLOWED_INTERPOLATORS[-1])


def main():

    input_parser = InputArgparser(
        description="Simulate stacks from obtained reconstruction. "
        "Script simulates/projects the slices at estimated positions "
        "within reconstructed volume. Ideally, if motion correction was "
        "correct, the resulting stack of such obtained projected slices, "
        "corresponds to the originally acquired (motion corrupted) data.",
    )
    input_parser.add_filenames(required=True)
    input_parser.add_filenames_masks()
    input_parser.add_dir_input_mc(required=True)
    input_parser.add_reconstruction(required=True)
    input_parser.add_dir_output(required=True)
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_prefix_output(default="Simulated_")
    input_parser.add_option(
        option_string="--copy-data",
        type=int,
        help="Turn on/off copying of original data (including masks) to "
        "output folder.",
        default=0)
    input_parser.add_option(
        option_string="--reconstruction-mask",
        type=str,
        help="If given, reconstruction image mask is propagated to "
        "simulated stack(s) of slices as well",
        default=None)
    input_parser.add_interpolator(
        option_string="--interpolator-mask",
        help="Choose the interpolator type to propagate the reconstruction "
        "mask (%s)." % (INTERPOLATOR_TYPES),
        default="NearestNeighbor")
    input_parser.add_log_config(default=0)
    input_parser.add_verbose(default=0)
    input_parser.add_slice_thicknesses(default=None)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    if args.interpolator_mask not in ALLOWED_INTERPOLATORS:
        raise IOError(
            "Unknown interpolator provided. Possible choices are %s" % (
                INTERPOLATOR_TYPES))

    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))

    # Read motion corrected data
    data_reader = dr.MultipleImagesReader(
        file_paths=args.filenames,
        file_paths_masks=args.filenames_masks,
        suffix_mask=args.suffix_mask,
        dir_motion_correction=args.dir_input_mc,
        stacks_slice_thicknesses=args.slice_thicknesses,
    )
    data_reader.read_data()
    stacks = data_reader.get_data()

    reconstruction = st.Stack.from_filename(
        args.reconstruction, args.reconstruction_mask, extract_slices=False)

    linear_operators = lin_op.LinearOperators()

    for i, stack in enumerate(stacks):

        # initialize image data array(s)
        nda = np.zeros_like(sitk.GetArrayFromImage(stack.sitk))
        nda[:] = np.nan

        if args.reconstruction_mask:
            nda_mask = np.zeros_like(sitk.GetArrayFromImage(stack.sitk_mask))

        slices = stack.get_slices()
        kept_indices = [s.get_slice_number() for s in slices]

        # Fill stack information "as if slice was acquired consecutively"
        # Therefore, simulated stack slices correspond to acquired slices
        # (in case motion correction was correct)
        for j in range(nda.shape[0]):
            if j in kept_indices:
                index = kept_indices.index(j)
                simulated_slice = linear_operators.A(
                    reconstruction,
                    slices[index],
                    interpolator_mask=args.interpolator_mask
                )
                nda[j, :, :] = sitk.GetArrayFromImage(simulated_slice.sitk)

                if args.reconstruction_mask:
                    nda_mask[j, :, :] = sitk.GetArrayFromImage(
                        simulated_slice.sitk_mask)

        # Create nifti image with same image header as original stack
        simulated_stack_sitk = sitk.GetImageFromArray(nda)
        simulated_stack_sitk.CopyInformation(stack.sitk)

        if args.reconstruction_mask:
            simulated_stack_sitk_mask = sitk.GetImageFromArray(nda_mask)
            simulated_stack_sitk_mask.CopyInformation(stack.sitk_mask)
        else:
            simulated_stack_sitk_mask = None

        simulated_stack = st.Stack.from_sitk_image(
            image_sitk=simulated_stack_sitk,
            image_sitk_mask=simulated_stack_sitk_mask,
            filename=args.prefix_output + stack.get_filename(),
            extract_slices=False,
            slice_thickness=stack.get_slice_thickness(),
        )

        if args.verbose:
            sitkh.show_stacks([
                stack, simulated_stack],
                segmentation=stack)

        simulated_stack.write(
            args.dir_output,
            write_mask=False,
            write_slices=False,
            suffix_mask=args.suffix_mask)

        if args.copy_data:
            stack.write(
                args.dir_output,
                write_mask=True,
                write_slices=False,
                suffix_mask="_mask")

    return 0


if __name__ == '__main__':
    main()
