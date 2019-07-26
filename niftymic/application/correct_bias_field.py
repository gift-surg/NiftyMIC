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
        description="Perform Bias Field correction using N4ITK.",
    )
    input_parser.add_filename(required=True)
    input_parser.add_output(required=True)
    input_parser.add_filename_mask()
    input_parser.add_option(
        option_string="--convergence-threshold",
        type=float,
        help="Specify the convergence threshold.",
        default=1e-6,
    )
    input_parser.add_option(
        option_string="--spline-order",
        type=int,
        help="Specify the spline order defining the bias field estimate.",
        default=3,
    )
    input_parser.add_option(
        option_string="--wiener-filter-noise",
        type=float,
        help="Specify the noise estimate defining the Wiener filter.",
        default=0.11,
    )
    input_parser.add_option(
        option_string="--bias-field-fwhm",
        type=float,
        help="Specify the full width at half maximum parameter characterizing "
        "the width of the Gaussian deconvolution.",
        default=0.15,
    )
    input_parser.add_log_config(default=1)
    input_parser.add_verbose(default=0)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    if np.alltrue([not args.output.endswith(t) for t in ALLOWED_EXTENSIONS]):
        raise ValueError(
            "output filename invalid; allowed extensions are: %s" %
            ", ".join(ALLOWED_EXTENSIONS))

    if args.log_config:
        input_parser.log_config(os.path.abspath(__file__))

    # Read data
    stack = st.Stack.from_filename(
        file_path=args.filename,
        file_path_mask=args.filename_mask,
        extract_slices=False,
    )

    # Perform Bias Field Correction
    ph.print_title("Perform Bias Field Correction")
    bias_field_corrector = n4itk.N4BiasFieldCorrection(
        stack=stack,
        use_mask=True if args.filename_mask is not None else False,
        convergence_threshold=args.convergence_threshold,
        spline_order=args.spline_order,
        wiener_filter_noise=args.wiener_filter_noise,
        bias_field_fwhm=args.bias_field_fwhm,
    )
    ph.print_info("N4ITK Bias Field Correction ... ", newline=False)
    bias_field_corrector.run_bias_field_correction()
    stack_corrected = bias_field_corrector.get_bias_field_corrected_stack()
    print("done")

    dw.DataWriter.write_image(stack_corrected.sitk, args.output)

    elapsed_time = ph.stop_timing(time_start)

    if args.verbose:
        ph.show_niftis([args.filename, args.output])

    ph.print_title("Summary")
    exe_file_info = os.path.basename(os.path.abspath(__file__)).split(".")[0]
    print("%s | Computational Time for Bias Field Correction: %s" % (
        exe_file_info, elapsed_time))

    return 0


if __name__ == '__main__':
    main()
