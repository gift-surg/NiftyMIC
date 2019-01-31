##
# \file evaluate_simulated_stack_similarity.py
# \brief      Script to evaluate the similarity of simulated stack from
#             obtained reconstruction against the original stack.
#
# This function takes the result of simulate_stacks_from_reconstruction.py as
# input.
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       November 2017
#

# Import libraries
import SimpleITK as sitk
import numpy as np
import os

import pysitk.python_helper as ph
from nsol.similarity_measures import SimilarityMeasures as \
    SimilarityMeasures

import niftymic.base.data_reader as dr
from niftymic.utilities.input_arparser import InputArgparser


def main():

    # Read input
    input_parser = InputArgparser(
        description="Script to evaluate the similarity of simulated stack "
        "from obtained reconstruction against the original stack. "
        "This function takes the result of "
        "simulate_stacks_from_reconstruction.py as input.",
    )
    input_parser.add_filenames(required=True)
    input_parser.add_filenames_masks()
    input_parser.add_dir_output(required=True)
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_measures(default=["NCC", "SSIM"])
    input_parser.add_option(
        option_string="--prefix-simulated",
        type=str,
        help="Specify the prefix of the simulated stacks to distinguish them "
        "from the original data.",
        default="Simulated_",
    )
    input_parser.add_option(
        option_string="--dir-input-simulated",
        type=str,
        help="Specify the directory where the simulated stacks are. "
        "If not given, it is assumed that they are in the same directory "
        "as the original ones.",
        default=None
    )
    input_parser.add_slice_thicknesses(default=None)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")

    # Read original data
    filenames_original = args.filenames
    data_reader = dr.MultipleImagesReader(
        file_paths=filenames_original,
        file_paths_masks=args.filenames_masks,
        suffix_mask=args.suffix_mask,
        stacks_slice_thicknesses=args.slice_thicknesses,
    )
    data_reader.read_data()
    stacks_original = data_reader.get_data()

    # Read data simulated from obtained reconstruction
    if args.dir_input_simulated is None:
        dir_input_simulated = os.path.dirname(filenames_original[0])
    else:
        dir_input_simulated = args.dir_input_simulated
    filenames_simulated = [
        os.path.join("%s", "%s%s") %
        (dir_input_simulated, args.prefix_simulated, os.path.basename(f))
        for f in filenames_original
    ]
    data_reader = dr.MultipleImagesReader(
        filenames_simulated, suffix_mask=args.suffix_mask)
    data_reader.read_data()
    stacks_simulated = data_reader.get_data()

    for i in range(len(stacks_original)):
        try:
            stacks_original[i].sitk - stacks_simulated[i].sitk
        except:
            raise IOError("Images '%s' and '%s' do not occupy the same space!"
                          % (filenames_original[i], filenames_simulated[i]))

    similarity_measures = {
        m: SimilarityMeasures.similarity_measures[m] for m in args.measures
    }
    similarities = np.zeros(len(args.measures))

    for i in range(len(stacks_original)):
        nda_3D_original = sitk.GetArrayFromImage(stacks_original[i].sitk)
        nda_3D_simulated = sitk.GetArrayFromImage(stacks_simulated[i].sitk)
        nda_3D_mask = sitk.GetArrayFromImage(stacks_original[i].sitk_mask)

        path_to_file = os.path.join(
            args.dir_output, "Similarity_%s.txt" %
            stacks_original[i].get_filename())
        text = "# Similarity: %s vs %s (%s)." % (
            os.path.basename(filenames_original[i]),
            os.path.basename(filenames_simulated[i]), ph.get_time_stamp())
        text += "\n#\t" + ("\t").join(args.measures)
        text += "\n"
        ph.write_to_file(path_to_file, text, "w")
        for k in range(nda_3D_original.shape[0]):
            x_2D_original = nda_3D_original[k, :, :]
            x_2D_simulated = nda_3D_simulated[k, :, :]

            # zero slice, i.e. rejected during motion correction
            if np.abs(x_2D_simulated).sum() < 1e-6:
                x_2D_simulated[:] = np.nan
            x_2D_mask = nda_3D_mask[k, :, :]

            indices = np.where(x_2D_mask > 0)

            for m, measure in enumerate(args.measures):
                if len(indices[0]) > 0:
                    similarities[m] = similarity_measures[measure](
                        x_2D_original[indices], x_2D_simulated[indices])
                else:
                    similarities[m] = np.nan
            ph.write_array_to_file(path_to_file, similarities.reshape(1, -1))

    return 0


if __name__ == '__main__':
    main()
