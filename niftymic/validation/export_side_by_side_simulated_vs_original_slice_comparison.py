##
# \file export_side_by_side_simulated_vs_original_slice_comparison.py
# \brief      Script to generate a pdf holding all side-by-side comparisons.
#
# This function takes the result of simulate_stacks_from_reconstruction.py as
# input. The script relies on ImageMagick.
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       November 2017
#

# Import libraries
import SimpleITK as sitk
import numpy as np
import natsort
import os
import re

import pysitk.python_helper as ph
from nsol.similarity_measures import SimilarityMeasures as \
    SimilarityMeasures

import niftymic.base.data_reader as dr
from niftymic.utilities.input_arparser import InputArgparser
from niftymic.definitions import DIR_TMP


def export_pdf_comparison(nda_original, nda_projected, path_to_file, extension="png"):
    dir_tmp = os.path.join(DIR_TMP, "ImageMagick")
    ph.clear_directory(dir_tmp, verbose=False)
    for k in range(nda_original.shape[0]):
        ctr = k+1
        # Export as individual image side-by-side
        export_image_side_by_side(
            nda_left=nda_original[k, :, :],
            nda_right=nda_projected[k, :, :],
            label_left="original",
            label_right="projected",
            path_to_file=os.path.join(
                dir_tmp, "%03d.%s" % (ctr, extension)),
            ctr=ctr,
            extension=extension,
        )
    # Combine all side-by-side images to single pdf
    export_pdf_from_side_by_side_images(
        dir_tmp, path_to_file, extension=extension)
    ph.print_info("Side-by-side comparison exported to '%s'" % path_to_file)


def rescale_image(path_to_file, scale=3):

    factor = scale * 100
    cmd_args = []
    cmd_args.append("%s" % path_to_file)
    cmd_args.append("-resize %dx%d%%\\!" % (factor, factor))
    cmd = "convert %s %s" % ((" ").join(cmd_args), path_to_file)
    ph.execute_command(cmd, verbose=False)


def export_image_side_by_side(
        nda_left,
        nda_right,
        label_left,
        label_right,
        path_to_file,
        ctr,
        extension,
        border=10,
        background="black",
        fill_ctr="orange",
        fill_label="white",
        font="Arial",
        pointsize=12,
):

    dir_output = os.path.join(DIR_TMP, "ImageMagick", "side-by-side")
    ph.clear_directory(dir_output, verbose=False)

    path_to_left = os.path.join(dir_output, "left.%s" % extension)
    path_to_right = os.path.join(dir_output, "right.%s" % extension)

    ph.write_image(nda_left, path_to_left, verbose=False)
    ph.write_image(nda_right, path_to_right, verbose=False)

    rescale_image(path_to_left)
    rescale_image(path_to_right)

    cmd_args = []
    cmd_args.append("-geometry +%d+%d" % (border, border))
    cmd_args.append("-background %s" % background)
    cmd_args.append("-font %s" % font)
    cmd_args.append("-pointsize %s" % pointsize)
    cmd_args.append("-fill %s" % fill_ctr)
    cmd_args.append("-gravity SouthWest -draw \"text 0,0 '%d'\"" % ctr)
    cmd_args.append("-fill %s" % fill_label)
    cmd_args.append("-label '%s' %s" % (label_left, path_to_left))
    cmd_args.append("-label '%s' %s" % (label_right, path_to_right))
    cmd_args.append("%s" % path_to_file)
    cmd = "montage %s" % (" ").join(cmd_args)
    ph.execute_command(cmd, verbose=False)


def export_pdf_from_side_by_side_images(directory, path_to_file, extension):
    pattern = "[a-zA-Z0-9_]+[.]%s" % extension
    p = re.compile(pattern)

    files = [os.path.join(directory, f)
             for f in os.listdir(directory) if p.match(f)]
    files = natsort.natsorted(files, key=lambda y: y.lower())
    cmd = "convert %s %s" % ((" ").join(files), path_to_file)
    ph.execute_command(cmd, verbose=False)


def main():

    # Read input
    input_parser = InputArgparser(
        description="Script to evaluate the similarity of simulated stack "
        "from obtained reconstruction against the original stack. "
        "This function takes the result of "
        "simulate_stacks_from_reconstruction.py as input.",
    )
    input_parser.add_filenames(required=True)
    input_parser.add_dir_output(required=True)
    input_parser.add_suffix_mask(default="_mask")
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

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")

    # Read original data
    filenames_original = args.filenames
    data_reader = dr.MultipleImagesReader(
        filenames_original, suffix_mask=args.suffix_mask)
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

    ph.create_directory(args.dir_output)

    for i in range(len(stacks_original)):
        try:
            stacks_original[i].sitk - stacks_simulated[i].sitk
        except:
            raise IOError("Images '%s' and '%s' do not occupy the same space!"
                          % (filenames_original[i], filenames_simulated[i]))

    intensity_max = 255
    intensity_min = 0
    for i in range(len(stacks_original)):
        nda_3D_original = sitk.GetArrayFromImage(stacks_original[i].sitk)
        nda_3D_simulated = sitk.GetArrayFromImage(stacks_simulated[i].sitk)

        # Scale uniformly between 0 and 255 according to the simulated stack
        # for export to png
        scale = np.max(nda_3D_simulated)
        nda_3D_original = intensity_max * nda_3D_original / scale
        nda_3D_simulated = intensity_max * nda_3D_simulated / scale

        nda_3D_simulated = np.clip(
            nda_3D_simulated, intensity_min, intensity_max)
        nda_3D_original = np.clip(
            nda_3D_original, intensity_min, intensity_max)

        filename = stacks_original[i].get_filename()
        path_to_file = os.path.join(args.dir_output, "%s.pdf" % filename)

        # Export side-by-side comparison of stack to pdf
        export_pdf_comparison(nda_3D_original, nda_3D_simulated, path_to_file)

if __name__ == '__main__':
    main()
