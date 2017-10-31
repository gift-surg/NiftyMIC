##
# \file mulitply_stack_with_mask.py
# \brief      Script to stack/reconstruction with multiply template mask.
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
import re

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

from niftymic.definitions import DIR_TMP
from niftymic.definitions import ALLOWED_EXTENSIONS
import niftymic.base.data_reader as dr
import niftymic.base.stack as st
from niftymic.utilities.input_arparser import InputArgparser
from niftymic.definitions import DIR_TEMPLATES


def main():

    input_parser = InputArgparser(
        description="Multiply stack/reconstruction with template mask.",
    )

    input_parser.add_filename(required=True)
    input_parser.add_option(
        option_string="--dir-input-templates",
        type=str,
        help="Input directory for templates",
        default=DIR_TEMPLATES),
    input_parser.add_option(
        option_string="--gestational-age",
        type=int,
        help="Gestational age of fetus to reconstruct",
        required=True,
        default=None)
    input_parser.add_verbose(default=1)
    input_parser.add_prefix_output(default="Masked_")
    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    template = os.path.join(
        args.dir_input_templates,
        "STA%d.nii.gz" % args.gestational_age)
    template_mask = os.path.join(
        args.dir_input_templates,
        "STA%d_mask.nii.gz" % args.gestational_age)

    stack = st.Stack.from_filename(args.filename, template_mask,
                                   extract_slices=False)
    stack_masked = stack.get_stack_multiplied_with_mask()
    stack_masked.set_filename(args.prefix_output + stack.get_filename())
    stack_masked.write(os.path.dirname(args.filename))

    if args.verbose:
        sitkh.show_stacks([stack, stack_masked], segmentation=stack)

if __name__ == '__main__':
    main()
