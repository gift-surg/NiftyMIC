#!/usr/bin/python

##
# \file runIntensityCorrection.py
# \brief      Script to correct for bias field. Based on N4ITK
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

import pythonhelper.PythonHelper as ph
import pythonhelper.SimpleITKHelper as sitkh

# Import modules
import volumetricreconstruction.base.DataReader as dr
import volumetricreconstruction.base.Stack as st
import volumetricreconstruction.registration.FLIRT as regflirt
import volumetricreconstruction.registration.NiftyReg as regniftyreg
import volumetricreconstruction.registration.RegistrationSimpleITK as regsitk
import volumetricreconstruction.preprocessing.IntensityCorrection as ic
from volumetricreconstruction.utilities.InputArparser import InputArgparser


if __name__ == '__main__':

    time_start = ph.start_timing()

    np.set_printoptions(precision=3)

    input_parser = InputArgparser(
        description="Run intensity correction",
        prog="python " + os.path.basename(__file__),
    )
    input_parser.add_filenames(required=True)
    input_parser.add_dir_output(required=True)
    input_parser.add_reference(required=True)
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_reference_mask()
    input_parser.add_prefix_output(default="IC_")
    input_parser.add_log_script_execution(default=1)
    input_parser.add_option(
        option_string="--registration",
        type=int,
        help="Turn on/off registration from image to reference prior to "
        "intensity correction.",
        default=0)
    input_parser.add_verbose(default=0)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # Write script execution call
    if args.log_script_execution:
        input_parser.write_performed_script_execution(
            os.path.abspath(__file__))

    # Read data
    data_reader = dr.MultipleImagesReader(
        args.filenames, suffix_mask=args.suffix_mask)
    data_reader.read_data()
    stacks = data_reader.get_stacks()

    reference = st.Stack.from_filename(
        file_path=args.reference,
        file_path_mask=args.reference_mask,
        extract_slices=False)

    # if args.verbose:
    #     stacks_vis = [s for s in stacks]
    #     stacks_vis.insert(0, reference)
    #     sitkh.show_stacks(stacks_vis)

    if args.registration:
        # registration = regniftyreg.RegAladin(
        registration = regflirt.FLIRT(
            moving=reference,
            registration_type="Rigid",
            use_fixed_mask=True,
            use_moving_mask=True,
            use_verbose=False,
        )

    # Perform Intensity Correction
    ph.print_title("Perform Intensity Correction")
    intensity_corrector = ic.IntensityCorrection(
        use_reference_mask=True,
        use_individual_slice_correction=False,
        prefix_corrected=args.prefix_output,
    )
    stacks_corrected = [None] * len(stacks)
    for i, stack in enumerate(stacks):
        if args.registration:
            ph.print_info("Image %d/%d: Registration ... "
                          % (i+1, len(stacks)), newline=False)
            registration.set_fixed(stack)
            registration.run_registration()
            transform_sitk = registration.get_registration_transform_sitk()
            stack.update_motion_correction(transform_sitk)
            print("done")

        ph.print_info("Image %d/%d: Intensity Correction ... "
                      % (i+1, len(stacks)), newline=False)

        ref = reference.get_resampled_stack(stack.sitk)
        ref = st.Stack.from_sitk_image(
            image_sitk=ref.sitk,
            image_sitk_mask=stack.sitk_mask*ref.sitk_mask,
            filename=reference.get_filename()
        )
        intensity_corrector.set_stack(stack)
        intensity_corrector.set_reference(ref)
        intensity_corrector.run_linear_intensity_correction()
        # intensity_corrector.run_affine_intensity_correction()
        stacks_corrected[i] = \
            intensity_corrector.get_intensity_corrected_stack()
        print("done")

        # Write Data
        stacks_corrected[i].write(
            args.dir_output, write_mask=True, suffix_mask=args.suffix_mask)

        if args.verbose:
            sitkh.show_stacks([
                reference, stacks_corrected[i],
                # stacks[i],
                ],
                segmentation=stacks_corrected[i])
            # ph.pause()

    # Write reference too (although not intensity corrected)
    reference.write(args.dir_output,
                    filename=args.prefix_output+reference.get_filename(),
                    write_mask=True, suffix_mask=args.suffix_mask)

    elapsed_time = ph.stop_timing(time_start)

    ph.print_title("Summary")
    print("Computational Time for Bias Field Correction(s): %s" %
          (elapsed_time))
