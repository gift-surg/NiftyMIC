#!/usr/bin/python

##
# \file registerToTemplateSpace.py
# \brief      Script to register the obtained reconstruction to a template
#             space.
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

import pysitk.PythonHelper as ph
import pysitk.SimpleITKHelper as sitkh

# Import modules
import niftymic.base.DataReader as dr
import niftymic.base.Stack as st
import niftymic.registration.FLIRT as regflirt
import niftymic.registration.NiftyReg as regniftyreg
import niftymic.registration.RegistrationSimpleITK as regsitk
import niftymic.preprocessing.IntensityCorrection as ic
from niftymic.utilities.InputArparser import InputArgparser

if __name__ == '__main__':

    time_start = ph.start_timing()

    np.set_printoptions(precision=3)

    input_parser = InputArgparser(
        description="Register an obtained reconstruction to a template space "
        "using rigid registration. "
        "The resulting registration can optionally be applied to previously "
        "obtained motion correction slice transforms so that a volumetric "
        "reconstruction is possible in the (standard anatomical) space "
        "defined by the template.",
        prog="python " + os.path.basename(__file__),
    )
    input_parser.add_option(
        option_string="--template",
        type=str,
        help="Template image used to perform reorientation.",
        required=True)
    input_parser.add_option(
        option_string="--reconstruction",
        type=str,
        help="Image which shall be registered to the template space.",
        required=True)
    input_parser.add_dir_input()
    input_parser.add_dir_output(required=True)
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_search_angle(default=180)
    input_parser.add_verbose(default=0)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")
    reconstruction = st.Stack.from_filename(args.reconstruction)
    template = st.Stack.from_filename(args.template)

    # -------------------Register Reconstruction to Template-------------------
    ph.print_title("Register Reconstruction to Template")

    # Define search angle ranges for FLIRT in all three dimensions
    search_angles = ["-searchr%s -%d %d" %
                     (x, args.search_angle, args.search_angle)
                     for x in ["x", "y", "z"]]
    search_angles = (" ").join(search_angles)

    registration = regflirt.FLIRT(
        fixed=reconstruction,
        moving=template,
        registration_type="Rigid",
        use_verbose=False,
        options=search_angles,
    )
    ph.print_info("Run Registration ... ", newline=False)
    registration.run_registration()
    print("done")
    transform_sitk = registration.get_registration_transform_sitk()

    # Apply rigidly transform to align reconstruction with template
    reconstruction_orient_sitk = sitkh.get_transformed_sitk_image(
        reconstruction.sitk, transform_sitk)
    reconstruction_orient = st.Stack.from_sitk_image(
        reconstruction_orient_sitk, filename=reconstruction.get_filename())

    # Resample reconstruction to template space
    reconstruction_orient = \
        reconstruction_orient.get_resampled_stack(template.sitk)
    reconstruction_orient.set_filename(
        reconstruction_orient.get_filename() + "ResamplingToTemplateSpace")

    # Write resampled reconstruction
    reconstruction_orient.write(args.dir_output)

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
        sitkh.show_stacks([template, reconstruction_orient])

    elapsed_time_total = ph.stop_timing(time_start)

    # Summary
    ph.print_title("Summary")
    print("Computational Time: %s" % (elapsed_time_total))
