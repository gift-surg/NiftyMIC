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
        description="Create and write random motion transforms.",
    )
    input_parser.add_dir_output(required=True)
    input_parser.add_option(
        option_string="--simulations", type=int, required=True)
    input_parser.add_option(option_string="--angle-max", default=10)
    input_parser.add_option(option_string="--translation-max", default=10)
    input_parser.add_option(option_string="--seed", type=int, default=1)
    input_parser.add_option(option_string="--dimension", type=int, default=3)
    input_parser.add_option(
        option_string="--write-settings", type=int, default=1)
    input_parser.add_prefix_output(default="Euler3DTransform_")
    input_parser.add_verbose(default=0)

    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    motion_simulator = ms.RandomRigidMotionSimulator(
        dimension=args.dimension,
        angle_max_deg=args.angle_max,
        translation_max=args.translation_max,
        verbose=args.verbose)
    motion_simulator.simulate_motion(
        seed=args.seed, simulations=args.simulations)

    prefix = "%sAngle%gTranslation%gSeed%d_" % (
        args.prefix_output, args.angle_max, args.translation_max, args.seed)
    prefix = prefix.replace(".", "p")
    motion_simulator.write_transforms_sitk(
        directory=args.dir_output,
        prefix_filename=prefix)

    return 0


if __name__ == '__main__':
    main()
