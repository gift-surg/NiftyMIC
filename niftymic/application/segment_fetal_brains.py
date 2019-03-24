##
# \file correct_bias_field.py
# \brief      Script to correct for bias field. Based on N4ITK
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       October 2017
#

# Import libraries
import numpy as np
import os

import niftymic.base.stack as st
import niftymic.base.data_writer as dw
import niftymic.utilities.n4_bias_field_correction as n4itk
import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh
from niftymic.utilities.input_arparser import InputArgparser

from niftymic.definitions import ALLOWED_EXTENSIONS


def main():
    time_start = ph.start_timing()

    np.set_printoptions(precision=3)

    input_parser = InputArgparser(
        description="Perform automatic brain masking using "
        "fetal_brain_seg (https://github.com/gift-surg/fetal_brain_seg). ",
    )
    input_parser.add_filenames(required=True)
    input_parser.add_filenames_masks(required=False)
    input_parser.add_dir_output(required=False)
    input_parser.add_verbose(default=0)
    input_parser.add_log_config(default=0)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    try:
        DIR_FETAL_BRAIN_SEG = os.environ["FETAL_BRAIN_SEG"]
    except KeyError as e:
        raise RuntimeError(
            "Environment variable FETAL_BRAIN_SEG is not specified. "
            "Specify the root directory of fetal_brain_seg "
            "(https://github.com/gift-surg/fetal_brain_seg) "
            "using "
            "'export FETAL_BRAIN_SEG=path_to_fetal_brain_seg_dir' "
            "(in bashrc).")

    if args.filenames_masks is None and args.dir_output is None:
        raise IOError("Either --filenames-masks or --dir-output must be set")

    if args.dir_output is not None:
        args.filenames_masks = [
            os.path.join(args.dir_output, os.path.basename(f))
            for f in args.filenames
        ]

    if len(args.filenames) != len(args.filenames_masks):
        raise IOError("Number of filenames and filenames-masks must match")

    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))

    cd_fetal_brain_seg = "cd %s" % DIR_FETAL_BRAIN_SEG

    for f, m in zip(args.filenames, args.filenames_masks):

        if not ph.file_exists(f):
            raise IOError("File '%s' does not exist" % f)

        # use absolute path for input image
        f = os.path.abspath(f)

        # use absolute path for output image
        dir_output = os.path.dirname(m)
        if not os.path.isabs(dir_output):
            dir_output = os.path.realpath(
                os.path.join(os.environ["PWD"], dir_output))
            m = os.path.join(dir_output, os.path.basename(m))

        ph.create_directory(dir_output)

        # Change to root directory of fetal_brain_seg
        cmds = [cd_fetal_brain_seg]

        # Run masking independently (Takes longer but ensures that it does
        # not terminate because of provided 'non-brain images')
        cmd_args = ["python fetal_brain_seg.py"]
        cmd_args.append("--input_names '%s'" % f)
        cmd_args.append("--segment_output_names '%s'" % m)
        cmds.append(" ".join(cmd_args))

        # Execute both steps
        cmd = " && ".join(cmds)
        flag = ph.execute_command(cmd)

        if flag != 0:
            ph.print_warning(
                "Error using fetal_brain_seg. \n"
                "Execute '%s' for further investigation" %
                cmd)

        ph.print_info("Fetal brain segmentation written to '%s'" % m)

        if args.verbose:
            ph.show_nifti(f, segmentation=m)

    return 0


if __name__ == '__main__':
    main()
