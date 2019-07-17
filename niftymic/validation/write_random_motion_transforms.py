##
# \file write_random_motion_transforms.py
# \brief      Create and write random motion transforms
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       November 2017
#

import numpy as np

from niftymic.utilities.input_arparser import InputArgparser
import niftymic.validation.motion_simulator as ms


def main():
    input_parser = InputArgparser(
        description="Create and write random rigid motion transformations. "
        "Simulated transformations are exported as (Simple)ITK transforms. ",
    )
    input_parser.add_dir_output(required=True)
    input_parser.add_option(
        option_string="--simulations",
        type=int,
        required=True,
        help="Number of simulated motion transformations."
    )
    input_parser.add_option(
        option_string="--angle-max",
        default=10,
        help="random angles (in degree) are drawn such "
        "that |angle| <= angle_max."
    )
    input_parser.add_option(
        option_string="--translation-max",
        default=10,
        help="random translations (in millimetre) are drawn such "
        "that |translation| <= translation_max."
    )
    input_parser.add_option(
        option_string="--seed",
        type=int,
        default=1,
        help="Seed for pseudo-random data generation"
    )
    input_parser.add_option(
        option_string="--dimension",
        type=int,
        default=3,
        help="Spatial dimension for transformations."
    )
    input_parser.add_prefix_output(default="EulerTransform_slice")
    input_parser.add_verbose(default=1)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    motion_simulator = ms.RandomRigidMotionSimulator(
        dimension=args.dimension,
        angle_max_deg=args.angle_max,
        translation_max=args.translation_max,
        verbose=args.verbose)
    motion_simulator.simulate_motion(
        seed=args.seed,
        simulations=args.simulations,
    )

    motion_simulator.write_transforms_sitk(
        directory=args.dir_output,
        prefix_filename=args.prefix_output,
    )

    return 0


if __name__ == '__main__':
    main()
