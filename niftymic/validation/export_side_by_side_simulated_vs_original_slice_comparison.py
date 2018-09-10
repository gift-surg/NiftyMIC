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


##
# Export a side-by-side comparison to a (pdf) file
# \date       2017-11-28 23:28:12+0000
#
# \param      nda_original   numpy data 3D array of original data
# \param      nda_projected  numpy data 3D array of projected/simulated data
# \param      path_to_file   path to file, string
# \param      resize         factor to resize images (otherwise they
#                            might be very small depending on the FOV)
# \param      extension      extension of images produced for tmp results.
#
def export_comparison_to_file(nda_original,
                              nda_projected,
                              path_to_file,
                              resize,
                              extension="png"):
    dir_tmp = os.path.join(DIR_TMP, "ImageMagick")
    ph.clear_directory(dir_tmp, verbose=False)
    for k in range(nda_original.shape[0]):
        ctr = k + 1

        # Export as individual image side-by-side
        _export_image_side_by_side(
            nda_left=nda_original[k, :, :],
            nda_right=nda_projected[k, :, :],
            label_left="original",
            label_right="projected",
            path_to_file=os.path.join(
                dir_tmp, "%03d.%s" % (ctr, extension)),
            ctr=ctr,
            resize=resize,
            extension=extension,
        )

    # Combine all side-by-side images to single pdf
    _export_pdf_from_side_by_side_images(
        dir_tmp, path_to_file, extension=extension)
    ph.print_info("Side-by-side comparison exported to '%s'" % path_to_file)

    # Delete tmp directory
    ph.delete_directory(dir_tmp, verbose=False)


##
# Export a single side-by-side comparison of two images
# \date       2017-11-28 23:30:23+0000
#
def _export_image_side_by_side(
        nda_left,
        nda_right,
        label_left,
        label_right,
        path_to_file,
        ctr,
        resize,
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

    nda_left = np.round(np.array(nda_left)).astype(np.uint8)
    nda_right = np.round(np.array(nda_right)).astype(np.uint8)
    ph.write_image(nda_left, path_to_left, verbose=False)
    ph.write_image(nda_right, path_to_right, verbose=False)

    _resize_image(path_to_left, resize=resize)
    _resize_image(path_to_right, resize=resize)

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


##
# Resize image
# \date       2017-11-28 23:31:07+0000
#
def _resize_image(path_to_file, resize):

    factor = resize * 100
    cmd_args = []
    cmd_args.append("%s" % path_to_file)
    cmd_args.append("-resize %dx%d%%\\!" % (factor, factor))
    cmd = "convert %s %s" % ((" ").join(cmd_args), path_to_file)
    ph.execute_command(cmd, verbose=False)


##
# Create single pdf from multiple side-by-side (png) images
# \date       2017-11-28 23:33:46+0000
#
# \param      directory     Path to directory with side-by-side png images
# \param      path_to_file  Path to combined pdf result
# \param      extension     The extension
#
def _export_pdf_from_side_by_side_images(directory, path_to_file, extension):

    # Read all sidy-by-side (png) images in directory
    pattern = "[a-zA-Z0-9_]+[.]%s" % extension
    p = re.compile(pattern)
    files = [os.path.join(directory, f)
             for f in os.listdir(directory) if p.match(f)]

    # Convert consecutive sequence of images into single pdf
    files = natsort.natsorted(files, key=lambda y: y.lower())
    cmd = "convert %s %s" % ((" ").join(files), path_to_file)
    ph.execute_command(cmd, verbose=False)


def main():

    input_parser = InputArgparser(
        description="Script to export a side-by-side comparison of originally "
        "acquired and simulated/projected slice given the estimated "
        "volumetric reconstruction."
        "This function takes the result of "
        "simulate_stacks_from_reconstruction.py as input.",
    )
    input_parser.add_filenames(required=True)
    input_parser.add_dir_output(required=True)
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
    input_parser.add_option(
        option_string="--resize",
        type=float,
        help="Factor to resize images (otherwise they might be very small "
        "depending on the FOV)",
        default=3)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")

    # Read original data
    filenames_original = args.filenames
    data_reader = dr.MultipleImagesReader(filenames_original)
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
    data_reader = dr.MultipleImagesReader(filenames_simulated)
    data_reader.read_data()
    stacks_simulated = data_reader.get_data()

    ph.create_directory(args.dir_output)

    for i in range(len(stacks_original)):
        try:
            stacks_original[i].sitk - stacks_simulated[i].sitk
        except:
            raise IOError("Images '%s' and '%s' do not occupy the same space!"
                          % (filenames_original[i], filenames_simulated[i]))

    # ---------------------Create side-by-side comparisons---------------------
    ph.print_title("Create side-by-side comparisons")
    intensity_max = 255
    intensity_min = 0
    for i in range(len(stacks_original)):
        ph.print_subtitle("Stack %d/%d" % (i + 1, len(stacks_original)))
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

        # Export side-by-side comparison of each stack to a pdf file
        export_comparison_to_file(
            nda_3D_original, nda_3D_simulated, path_to_file,
            resize=args.resize)

if __name__ == '__main__':
    main()
