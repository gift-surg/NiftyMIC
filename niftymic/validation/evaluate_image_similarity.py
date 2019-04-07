##
# \file evaluate_image_similarity.py
# \brief      Evaluate similarity to a reference of one or more images
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Feb 1
#

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

import nsol.observer as obs
from nsol.similarity_measures import SimilarityMeasures as \
    SimilarityMeasures

import niftymic.base.stack as st
import niftymic.base.data_reader as dr
import niftymic.registration.niftyreg as regniftyreg
from niftymic.utilities.input_arparser import InputArgparser

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh


def main():

    # Set print options
    np.set_printoptions(precision=3)
    pd.set_option('display.width', 1000)

    input_parser = InputArgparser(
        description=".",
    )
    input_parser.add_filenames(required=True)
    input_parser.add_reference(required=True)
    input_parser.add_reference_mask()
    input_parser.add_dir_output(required=False)
    input_parser.add_measures(
        default=["PSNR", "RMSE", "MAE", "SSIM", "NCC", "NMI"])
    input_parser.add_verbose(default=0)
    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    ph.print_title("Image similarity")
    data_reader = dr.MultipleImagesReader(args.filenames)
    data_reader.read_data()
    stacks = data_reader.get_data()

    reference = st.Stack.from_filename(args.reference, args.reference_mask)

    for stack in stacks:
        try:
            stack.sitk - reference.sitk
        except RuntimeError as e:
            raise IOError(
                "All provided images must be at the same image space")

    x_ref = sitk.GetArrayFromImage(reference.sitk)

    if args.reference_mask is None:
        indices = np.where(x_ref != np.inf)
    else:
        x_ref_mask = sitk.GetArrayFromImage(reference.sitk_mask)
        indices = np.where(x_ref_mask > 0)

    measures_dic = {
        m: lambda x, m=m:
        SimilarityMeasures.similarity_measures[m](
            x[indices], x_ref[indices])
        # SimilarityMeasures.similarity_measures[m](x, x_ref)
        for m in args.measures
    }

    observer = obs.Observer()
    observer.set_measures(measures_dic)
    for stack in stacks:
        nda = sitk.GetArrayFromImage(stack.sitk)
        observer.add_x(nda)

    if args.verbose:
        stacks_comparison = [s for s in stacks]
        stacks_comparison.insert(0, reference)
        sitkh.show_stacks(
            stacks_comparison,
            segmentation=reference,
        )

    observer.compute_measures()
    measures = observer.get_measures()

    # Store information in array
    error = np.zeros((len(stacks), len(measures)))
    cols = measures
    rows = []
    for i_stack, stack in enumerate(stacks):
        error[i_stack, :] = np.array([measures[m][i_stack] for m in measures])
        rows.append(stack.get_filename())

    header = "# Ref: %s, Ref-Mask: %d, %s \n" % (
        reference.get_filename(),
        args.reference_mask is None,
        ph.get_time_stamp(),
    )
    header += "# %s\n" % ("\t").join(measures)

    path_to_file_filenames = os.path.join(
        args.dir_output, "filenames.txt")
    path_to_file_similarities = os.path.join(
        args.dir_output, "similarities.txt")

    # Write to files
    ph.write_to_file(path_to_file_similarities, header)
    ph.write_array_to_file(
        path_to_file_similarities, error, verbose=False)
    text = header
    text += "%s\n" %"\n".join(rows)
    ph.write_to_file(path_to_file_filenames, text)

    # Print to screen
    ph.print_subtitle("Computed Similarities")
    df = pd.DataFrame(error, rows, cols)
    print(df)

    return 0


if __name__ == '__main__':
    main()
