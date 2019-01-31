##
# \file multiply.py
# \brief      Script multiply images with each other.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       October 2017
#

import os
import SimpleITK as sitk

import pysitk.python_helper as ph

import niftymic.base.data_writer as dw
from niftymic.utilities.input_arparser import InputArgparser


def main():

    input_parser = InputArgparser(
        description="Multiply images. "
        "Pixel type is determined by first given image.",
    )

    input_parser.add_filenames(required=True)
    input_parser.add_output(required=True)
    input_parser.add_verbose(default=0)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    if len(args.filenames) < 2:
        raise IOError("At least two images must be provided")

    out_sitk = sitk.ReadImage(args.filenames[0])
    for f in args.filenames[1:]:
        im_sitk = sitk.Cast(sitk.ReadImage(f), out_sitk.GetPixelIDValue())
        out_sitk = out_sitk * im_sitk

    dw.DataWriter.write_image(out_sitk, args.output)

    if args.verbose:
        args.filenames.insert(0, args.output)
        ph.show_niftis(args.filenames)

if __name__ == '__main__':
    main()
