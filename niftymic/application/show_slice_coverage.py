##
# \file show_slice_coverage.py
# \brief      Script to show slice coverage available over reconstruction
#             space.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Feb 2019
#

import os
import SimpleITK as sitk

import pysitk.python_helper as ph

import niftymic.base.data_reader as dr
import niftymic.base.data_writer as dw
import niftymic.validation.slice_coverage as sc
from niftymic.utilities.input_arparser import InputArgparser


def main():

    input_parser = InputArgparser(
        description="Show data/slice coverage over specified reconstruction "
        "space.",
    )

    input_parser.add_filenames(required=True)
    input_parser.add_reconstruction_space(required=True)
    input_parser.add_output(required=True)
    input_parser.add_dir_input_mc()
    input_parser.add_slice_thicknesses()
    input_parser.add_verbose(default=0)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    data_reader = dr.MultipleImagesReader(
        file_paths=args.filenames,
        dir_motion_correction=args.dir_input_mc,
        stacks_slice_thicknesses=args.slice_thicknesses,
    )
    data_reader.read_data()
    stacks = data_reader.get_data()

    reconstruction_space_sitk = sitk.ReadImage(args.reconstruction_space)
    slice_coverage = sc.SliceCoverage(
        stacks=stacks,
        reconstruction_sitk=reconstruction_space_sitk,
    )
    slice_coverage.run()

    coverage_sitk = slice_coverage.get_coverage_sitk()

    dw.DataWriter.write_mask(coverage_sitk, args.output)

    if args.verbose:
        niftis = [
            args.reconstruction_space,
            args.output,
        ]
        ph.show_niftis(niftis)

if __name__ == '__main__':
    main()
