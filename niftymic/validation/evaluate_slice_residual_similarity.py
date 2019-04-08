##
# \file evaluate_slice_residual_similarity.py
# \brief      Evaluate similarity to a reference of one or more images
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Feb 2018
#

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.stack as st
import niftymic.base.data_reader as dr
import niftymic.utilities.intensity_correction as ic
import niftymic.registration.niftyreg as regniftyreg
import niftymic.validation.residual_evaluator as res_ev
from niftymic.utilities.input_arparser import InputArgparser


def main():

    time_start = ph.start_timing()

    # Set print options
    np.set_printoptions(precision=3)
    pd.set_option('display.width', 1000)

    input_parser = InputArgparser(
        description=".",
    )
    input_parser.add_filenames()
    input_parser.add_filenames_masks()
    input_parser.add_dir_input_mc()
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_reference(required=True)
    input_parser.add_reference_mask()
    input_parser.add_dir_output(required=False)
    input_parser.add_log_config(default=1)
    input_parser.add_measures(
        default=["PSNR", "MAE", "RMSE", "SSIM", "NCC", "NMI"])
    input_parser.add_verbose(default=0)
    input_parser.add_target_stack(default=None)
    input_parser.add_intensity_correction(default=1)
    input_parser.add_slice_thicknesses(default=None)
    input_parser.add_option(
        option_string="--use-reference-mask", type=int, default=1)
    input_parser.add_option(
        option_string="--use-slice-masks", type=int, default=1)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")

    data_reader = dr.MultipleImagesReader(
        file_paths=args.filenames,
        file_paths_masks=args.filenames_masks,
        suffix_mask=args.suffix_mask,
        dir_motion_correction=args.dir_input_mc,
        stacks_slice_thicknesses=args.slice_thicknesses,
    )

    data_reader.read_data()
    stacks = data_reader.get_data()
    ph.print_info("%d input stacks read for further processing" % len(stacks))

    # Specify target stack for intensity correction and reconstruction space
    if args.target_stack is None:
        target_stack_index = 0
    else:
        filenames = ["%s.nii.gz" % s.get_filename() for s in stacks]
        filename_target_stack = os.path.basename(args.target_stack)
        try:
            target_stack_index = filenames.index(filename_target_stack)
        except ValueError as e:
            raise ValueError(
                "--target-stack must correspond to an image as provided by "
                "--filenames")

    # ---------------------------Intensity Correction--------------------------
    if args.intensity_correction:
        ph.print_title("Intensity Correction")
        intensity_corrector = ic.IntensityCorrection()
        intensity_corrector.use_individual_slice_correction(False)
        intensity_corrector.use_stack_mask(True)
        intensity_corrector.use_reference_mask(True)
        intensity_corrector.use_verbose(False)

        for i, stack in enumerate(stacks):
            if i == target_stack_index:
                ph.print_info("Stack %d (%s): Reference image. Skipped." % (
                    i + 1, stack.get_filename()))
                continue
            else:
                ph.print_info("Stack %d (%s): Intensity Correction ... " % (
                    i + 1, stack.get_filename()), newline=False)
            intensity_corrector.set_stack(stack)
            intensity_corrector.set_reference(
                stacks[target_stack_index].get_resampled_stack(
                    resampling_grid=stack.sitk,
                    interpolator="NearestNeighbor",
                ))
            intensity_corrector.run_linear_intensity_correction()
            stacks[i] = intensity_corrector.get_intensity_corrected_stack()
            print("done (c1 = %g) " %
                  intensity_corrector.get_intensity_correction_coefficients())

    # ----------------------- Slice Residual Similarity -----------------------
    reference = st.Stack.from_filename(args.reference, args.reference_mask)

    ph.print_title("Slice Residual Similarity")
    residual_evaluator = res_ev.ResidualEvaluator(
        stacks=stacks,
        reference=reference,
        measures=args.measures,
        use_reference_mask=args.use_reference_mask,
        use_slice_masks=args.use_slice_masks,
    )
    residual_evaluator.compute_slice_projections()
    residual_evaluator.evaluate_slice_similarities()
    residual_evaluator.write_slice_similarities(args.dir_output)

    elapsed_time = ph.stop_timing(time_start)
    ph.print_title("Summary")
    print("Computational Time for Slice Residual Evaluation: %s" %
          (elapsed_time))

    return 0


if __name__ == '__main__':
    main()
