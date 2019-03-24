##
# \file input_argparser.py
# \brief      Class holding a collection of possible arguments to parse for
#             scripts
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       August 2017
#

import os
import re
import six
import sys
import inspect
import argparse
import platform
import datetime

import pysitk.python_helper as ph
from nsol.similarity_measures import SimilarityMeasures as SimilarityMeasures
from nsol.loss_functions import LossFunctions as LossFunctions

import niftymic
from niftymic.definitions import ALLOWED_EXTENSIONS
from niftymic.definitions import ALLOWED_INTERPOLATORS
from niftymic.definitions import VIEWER_OPTIONS, V2V_METHOD_OPTIONS

# Allowed image types
IMAGE_TYPES = "(" + (", or ").join(ALLOWED_EXTENSIONS) + ")"
INTERPOLATOR_TYPES = "(%s, or %s)" % (
    (", ").join(ALLOWED_INTERPOLATORS[0:-1]), ALLOWED_INTERPOLATORS[-1])


##
# Class holding a collection of possible arguments to parse for scripts
# \date       2017-08-07 01:26:11+0100
#
class InputArgparser(object):

    def __init__(self,
                 description=None,
                 prog=None,
                 epilog="Author: Michael Ebner (michael.ebner.14@ucl.ac.uk)",
                 config_arg="--config"
                 ):

        config_helper = "Args that start with '--' (eg. --dir-output) " \
            "can also be set in a config file (specified via %s). " \
            "If an arg is specified in more than one place, then " \
            "commandline values override config file values which " \
            "override defaults." % (config_arg)

        kwargs = {}
        if description is not None:
            kwargs['description'] = "%s %s" % (description, config_helper)
        if prog is not None:
            kwargs['prog'] = prog
        if epilog is not None:
            kwargs['epilog'] = epilog

        self._parser = argparse.ArgumentParser(**kwargs)
        self._parser.add_argument(
            config_arg,
            help="Configuration file in JSON format.")
        self._config_arg = config_arg

    def get_parser(self):
        return self._parser

    def parse_args(self):

        # read config file if available
        if self._config_arg in sys.argv:
            self._parse_config_file()

        return self._parser.parse_args()

    def print_arguments(self, args, title="Configuration:"):
        ph.print_title(title)
        for arg in sorted(vars(args)):
            ph.print_info("%s: " % (arg), newline=False)
            vals = getattr(args, arg)

            if type(vals) is list:
                # print list element in new lines, unless only one entry in list
                # if len(vals) == 1:
                #     print(vals[0])
                # else:
                print("")
                for val in vals:
                    print("\t%s" % val)
            else:
                print(vals)
        print("\nNiftyMIC version: %s" % niftymic.__version__)
        ph.print_line_separator(add_newline=False)
        print("")

    ##
    # Writes a performed script execution.
    # \date       2018-01-16 16:05:53+0000
    #
    # \param      self    The object
    # \param      file    path to executed file obtained, e.g. via
    #                     os.path.abspath(__file__)
    # \param      prefix  filename prefix
    #
    def log_config(self,
                   file,
                   prefix="config"):

        # parser returns options with underscores, e.g. 'dir_output'
        dic_with_underscores = vars(self._parser.parse_args())

        # get output directory to write log/config file
        try:
            dir_output = dic_with_underscores["dir_output"]
        except KeyError:
            dir_output = os.path.dirname(dic_with_underscores["output"])

        # build output file name
        name = os.path.basename(file).split(".")[0]
        now = datetime.datetime.now()
        time_stamp = now.strftime("%Y%m%d-%H%M%S")
        path_to_config_file = os.path.join(
            dir_output,
            "%s_%s_%s.json" % (prefix, name, time_stamp))

        # exclude config file (as setting in parameters reflected anyway)
        dic_with_underscores.pop(re.sub("--", "", self._config_arg))

        # replace underscore by dashes for correct commandline parsing,
        # e.g. "dir-output" instead of "dir_output"
        dic = {
            re.sub("_", "-", k): v
            for k, v in six.iteritems(dic_with_underscores)}

        # add user info to config file
        try:
            login = os.getlogin()
        except OSError as e:
            login = "unknown_login"
        node = platform.node()
        info_args = []
        info_args.append("Python %s" % platform.python_version())
        info_args.append(platform.system())
        info_args.append(platform.release())
        info_args.append(platform.version())
        info_args.append(platform.machine())
        info_args.append(platform.processor())
        user = "%s @ %s (%s)" % (login, node, ", ".join(info_args))
        dic["user"] = user

        # add date of execution to config file
        dic["date"] = now.strftime("%Y-%m-%d %H:%M:%S")

        # add version number to config file
        dic["version"] = niftymic.__version__

        # write config file to output
        ph.write_dictionary_to_json(dic, path_to_config_file, verbose=True)

    def add_filename(
        self,
        option_string="--filename",
        type=str,
        help="Path to NIfTI file %s." % (IMAGE_TYPES),
        default=None,
        required=True,
    ):
        self._add_argument(dict(locals()))

    def add_filename_mask(
        self,
        option_string="--filename-mask",
        type=str,
        help="Path to NIfTI file mask %s." % (IMAGE_TYPES),
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_dir_input(
        self,
        option_string="--dir-input",
        type=str,
        help="Input directory with NIfTI files %s." % (IMAGE_TYPES),
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_dir_input_mc(
        self,
        option_string="--dir-input-mc",
        type=str,
        help="Input directory where transformation files (.tfm) for "
        "motion-corrected slices are stored.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_subfolder_motion_correction(
        self,
        option_string="--subfolder-motion-correction",
        type=str,
        help="Name of folder within output directory where all motion "
        "correction results are stored",
        default="motion_correction",
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_subfolder_comparison(
        self,
        option_string="--subfolder-comparison",
        type=str,
        help="Name of folder within output directory where all comparison "
        "results are stored",
        default="comparison",
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_dir_inputs(
        self,
        option_string="--dir-inputs",
        nargs="+",
        type=str,
        help="Input directories with NIfTI files %s." % (IMAGE_TYPES),
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_filenames(
        self,
        option_string="--filenames",
        nargs="+",
        help="Paths to NIfTI file images %s." % (IMAGE_TYPES),
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_filenames_masks(
        self,
        option_string="--filenames-masks",
        nargs="+",
        help="Paths to NIfTI file image masks %s." % (IMAGE_TYPES),
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_label(
        self,
        option_string="--label",
        type=str,
        help="Label for image given by filename.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_fixed(
        self,
        option_string="--fixed",
        type=str,
        help="Fixed image to be registered.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_fixed_mask(
        self,
        option_string="--fixed-mask",
        type=str,
        help="Fixed image mask to be registered.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_moving(
        self,
        option_string="--moving",
        nargs=None,
        type=str,
        help="Moving image to be registered.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_moving_mask(
        self,
        option_string="--moving-mask",
        type=str,
        help="Moving image mask to be registered.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_metric(
        self,
        option_string="--metric",
        type=str,
        help="Metric for image registration method.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_metric_radius(
        self,
        option_string="--metric-radius",
        type=int,
        help="Radius in case metric 'ANTSNeighborhoodCorrelation' is chosen.",
        default=10,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_labels(
        self,
        option_string="--labels",
        nargs="+",
        help="Labels for images given by filenames.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_image_selection(
        self,
        option_string="--image-selection",
        nargs="+",
        help="Specify image filenames without filename extension which will "
        "be used only. ",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_reference(
        self,
        option_string="--reference",
        type=str,
        help="Path to reference image file %s." % (IMAGE_TYPES),
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_reference_mask(
        self,
        option_string="--reference-mask",
        type=str,
        help="Path to reference mask image file %s." % (IMAGE_TYPES),
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_output(
        self,
        option_string="--output",
        type=str,
        help="Path to output image file %s." % (IMAGE_TYPES),
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_dir_output(
        self,
        option_string="--dir-output",
        type=str,
        help="Output directory.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_suffix_mask(
        self,
        option_string="--suffix-mask",
        type=str,
        help="Suffix used to associate a mask with an image. "
        "E.g. suffix_mask='_mask' means an existing "
        "image_i_mask.nii.gz represents the mask to "
        "image_i.nii.gz for all images image_i in the input "
        "directory.",
        default="_mask",
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_use_masks_srr(
        self,
        option_string="--use-masks-srr",
        type=int,
        help="Use masks in SRR step to confine volumetric reconstruction only "
        "to the masked slice regions.",
        default=1,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_outlier_rejection(
        self,
        option_string="--outlier-rejection",
        type=int,
        help="Turn on/off use of outlier rejection mechanism to eliminate "
        "misregistered slices.",
        default=0,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_use_robust_registration(
        self,
        option_string="--use-robust-registration",
        type=int,
        help="Turn on/off use of robust slice-to-volume registration.",
        default=0,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_boundary_stacks(
        self,
        option_string="--boundary-stacks",
        type=int,
        nargs="+",
        help="Specify boundary in i-, j- and k-direction in mm "
        "for cropping the given input stacks. "
        "Stack will be cropped to bounding box encompassing the mask plus "
        "the added boundary.",
        default=[0, 0, 0],
    ):
        self._add_argument(dict(locals()))

    def add_prefix_output(
        self,
        option_string="--prefix-output",
        type=str,
        help="Prefix for SRR output file name.",
        default="SRR_",
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_target_stack_index(
        self,
        option_string="--target-stack-index",
        type=int,
        help="Index of input image stack that defines physical space for SRR. "
        "First index is 0.",
        default=0,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_target_stack(
        self,
        option_string="--target-stack",
        type=str,
        help="Choose target stack for reconstruction/pre-processing %s." % (
            IMAGE_TYPES),
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_reconstruction(
        self,
        option_string="--reconstruction",
        type=str,
        help="Path to NIfTI file %s of the obtained volumetric reconstruction "
        % (IMAGE_TYPES),
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_reconstruction_space(
        self,
        option_string="--reconstruction-space",
        type=str,
        help="Path to NIfTI file %s which defines the physical space "
        "for the volumetric reconstruction/SRR." % (IMAGE_TYPES),
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_gestational_age(
        self,
        option_string="--gestational-age",
        type=int,
        help="Gestational age in weeks of the fetal brain to "
        "be reconstructed.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_search_angle(
        self,
        option_string="--search-angle",
        type=int,
        help="Maximum search angle to be used to find correct orientation.",
        default=180,
    ):
        self._add_argument(dict(locals()))

    def add_multiresolution(
        self,
        option_string="--multiresolution",
        type=int,
        help="Turn on/off multiresolution approach for motion correction.",
        default=0,
    ):
        self._add_argument(dict(locals()))

    def add_shrink_factors(
        self,
        option_string="--shrink-factors",
        type=int,
        nargs="+",
        help="Specify shrink factors for multiresolution approach.",
        default=[2, 1],
    ):
        self._add_argument(dict(locals()))

    def add_smoothing_sigmas(
        self,
        option_string="--smoothing-sigmas",
        type=float,
        nargs="+",
        help="Specify smoothing sigmas for multiresolution approach.",
        default=[1, 0],
    ):
        self._add_argument(dict(locals()))

    def add_two_step_cycles(
        self,
        option_string="--two-step-cycles",
        type=int,
        help="Number of two-step-cycles, i.e. number of "
        "Slice-to-Volume Registration and Super-Resolution Reconstruction "
        "cycles",
        default=3,
    ):
        self._add_argument(dict(locals()))

    def add_sigma(
        self,
        option_string="--sigma",
        type=float,
        help="Standard deviation for Scattered Data Approximation approach "
        "to reconstruct first estimate of HR volume from all 3D input stacks.",
        default=0.9,
    ):
        self._add_argument(dict(locals()))

    def add_minimizer(
        self,
        option_string="--minimizer",
        type=str,
        help="Choice of minimizer used for the inverse problem associated to "
        "the SRR. Possible choices are 'lsmr' or any solver in "
        "scipy.optimize.minimize like 'L-BFGS-B'. Note, in case of a chosen "
        "non-linear data loss only non-linear solvers like 'L-BFGS-B' are "
        "viable.",
        default="lsmr",
    ):
        self._add_argument(dict(locals()))

    def add_alpha(
        self,
        option_string="--alpha",
        type=float,
        help="Regularization parameter alpha to solve the Super-Resolution "
        "Reconstruction problem: SRR = argmin_x "
        "[0.5 * sum_k ||y_k - A_k x||^2 + alpha * R(x)].",
        default=0.03,
    ):
        self._add_argument(dict(locals()))

    def add_alpha_first(
        self,
        option_string="--alpha-first",
        type=float,
        help="Regularization parameter like 'alpha' but used for the first"
        "SRR step.",
        default=0.1,
    ):
        self._add_argument(dict(locals()))

    def add_threshold(
        self,
        option_string="--threshold",
        type=float,
        help="Threshold between 0 and 1 to detect misregistered slices based "
        "on NCC in final cycle.",
        default=0.7,
    ):
        self._add_argument(dict(locals()))

    def add_threshold_first(
        self,
        option_string="--threshold-first",
        type=float,
        help="Threshold between 0 and 1 to detect misregistered slices based "
        "on NCC in first cycle.",
        default=0.6,
    ):
        self._add_argument(dict(locals()))

    def add_s2v_smoothing(
        self,
        option_string="--s2v-smoothing",
        type=float,
        help="Value for Gaussian process parameter smoothing.",
        default=0.5,
    ):
        self._add_argument(dict(locals()))

    def add_interleave(
        self,
        option_string="--interleave",
        type=int,
        help="Interleave used for slice acquisition",
        default=2,
    ):
        self._add_argument(dict(locals()))

    def add_iter_max(
        self,
        option_string="--iter-max",
        type=int,
        help="Number of maximum iterations for the numerical solver.",
        default=10,
    ):
        self._add_argument(dict(locals()))

    def add_iter_max_first(
        self,
        option_string="--iter-max-first",
        type=int,
        help="Number of maximum iterations for the numerical solver like "
        "'iter-max' but used for the first SRR step",
        default=5,
    ):
        self._add_argument(dict(locals()))

    def add_rho(
        self,
        option_string="--rho",
        type=float,
        help="Regularization parameter for augmented Lagrangian term required "
        "by ADMM approach for TV regularization",
        default=0.5,
    ):
        self._add_argument(dict(locals()))

    def add_iterations(
        self,
        option_string="--iterations",
        type=int,
        help="Number of ADMM/Primal Dual iterations.",
        default=10,
    ):
        self._add_argument(dict(locals()))

    def add_tv_solver(
        self,
        option_string="--tv-solver",
        type=str,
        help="Type of TV solver. Either 'ADMM' or 'PD'.",
        default="PD",
    ):
        self._add_argument(dict(locals()))

    def add_data_loss(
        self,
        option_string="--data-loss",
        type=str,
        help="Loss function rho used for data term, i.e. rho((y_k - A_k x)^2) "
        "Possible choices are 'linear', 'soft_l1, 'huber', 'arctan' and "
        "'cauchy'.",
        default="linear",
    ):
        self._add_argument(dict(locals()))

    def add_data_loss_scale(
        self,
        option_string="--data-loss-scale",
        type=float,
        help="Value of soft margin between inlier and outlier residuals, "
        "default is 1.0. The loss function is evaluated as "
        "rho_(f2) = C**2 * rho(f2 / C**2), where C is data_loss_scale. "
        "This parameter has no effect with data_loss='linear', but for other "
        "loss values it is of crucial importance.",
        default=1,
    ):
        self._add_argument(dict(locals()))

    def add_pd_alg_type(
            self,
            option_string="--pd-alg-type",
            type=str,
            help="Algorithm used to dynamically update parameters for each "
            "iteration of the dual algorithm. "
            "Possible choices are 'ALG2', 'ALG2_AHMOD' and 'ALG3' as "
            "described in Chambolle et al., 2011",
            default="ALG2",
    ):
        self._add_argument(dict(locals()))

    def add_dilation_radius(
        self,
        option_string="--dilation-radius",
        type=int,
        help="Dilation radius in number of voxels used for segmentation "
        "propagation from target stack in case masks are not provided for all "
        "images.",
        default=3,
    ):
        self._add_argument(dict(locals()))

    def add_extra_frame_target(
        self,
        option_string="--extra-frame-target",
        type=float,
        help="Increase chosen target space uniformly in each direction by "
        "extra frame given in mm.",
        default=0,
    ):
        self._add_argument(dict(locals()))

    def add_bias_field_correction(
        self,
        option_string="--bias-field-correction",
        type=int,
        help="Turn on/off bias field correction step during data "
        "preprocessing.",
        default=0,
    ):
        self._add_argument(dict(locals()))

    def add_intensity_correction(
        self,
        option_string="--intensity-correction",
        type=int,
        help="Turn on/off linear intensity correction step during data "
        "preprocessing.",
        default=0,
    ):
        self._add_argument(dict(locals()))

    def add_isotropic_resolution(
        self,
        option_string="--isotropic-resolution",
        type=float,
        help="Specify isotropic resolution for obtained SRR volume. Default "
        "resolution is specified by in-plane resolution of chosen target "
        "stack.",
        default=None,
    ):
        self._add_argument(dict(locals()))

    def add_log_config(
        self,
        option_string="--log-config",
        type=int,
        help="Turn on/off configuration log of executed script.",
        default=0,
    ):
        self._add_argument(dict(locals()))

    def add_write_motion_correction(
        self,
        option_string="--write-motion-correction",
        type=int,
        help="Turn on/off functionality to write final result of motion "
        "correction. This includes the rigidly aligned stacks with their "
        "respective motion corrected individual slices and the overall "
        "transform applied to each individual slice.",
        default=0,
    ):
        self._add_argument(dict(locals()))

    def add_provide_comparison(
        self,
        option_string="--provide-comparison",
        type=int,
        help="Turn on/off functionality to create files "
        "allowing for a visual comparison between original "
        "data and the obtained SRR. A folder 'comparison' "
        "will be created in the output directory containing "
        "the obtained SRR along with the linearly resampled "
        "original data. An additional script "
        "'show_comparison.py' will be provided whose "
        "execution will open all images in ITK-Snap "
        "(http://www.itksnap.org/).",
        default=0,
    ):
        self._add_argument(dict(locals()))

    def add_verbose(
        self,
        option_string="--verbose",
        type=int,
        help="Turn on/off verbose output.",
        default=1,
    ):
        self._add_argument(dict(locals()))

    def add_option(
        self,
        option_string="--option",
        nargs=None,
        type=float,
        help="Add option.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_argument(
        self,
        *a,
        **k
    ):
        self._parser.add_argument(*a, **k)

    def add_psf_aware(
        self,
        option_string='--psf-aware',
        type=int,
        help="Turn on/off use of PSF-aware registration.",
        default=0,
    ):
        self._add_argument(dict(locals()))

    def add_stack_recon_range(
        self,
        option_string="--stack-recon-range",
        type=int,
        help="Number of components used for SRR.",
        default=15,
    ):
        self._add_argument(dict(locals()))

    def add_alphas(
        self,
        option_string="--alphas",
        nargs="+",
        type=float,
        help="Specify regularization parameters to be looped through.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_data_losses(
        self,
        option_string="--data-losses",
        nargs="+",
        help="Specify data losses to be looped through.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_data_loss_scales(
        self,
        option_string="--data-loss-scales",
        nargs="+",
        type=float,
        help="Specify data loss scales to be looped through.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_study_name(
        self,
        option_string="--study-name",
        type=str,
        help="Name of parameter study (no white spaces).",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_measures(
        self,
        option_string="--measures",
        type=str,
        nargs="+",
        help="Measures to be evaluated between reference (if given) and "
        "reconstruction %s. " % ("(" + (", ").join(
            SimilarityMeasures.similarity_measures.keys()) + ")"),
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_reconstruction_type(
        self,
        option_string="--reconstruction-type",
        type=str,
        help="Define reconstruction type. Allowed values are "
        "'TK0L2', 'TK1L2', 'TVL2', and 'HuberL2'.",
        default="TVL1",
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_interpolator(
        self,
        option_string="--interpolator",
        type=str,
        help="Choose type of interpolator %s." % (INTERPOLATOR_TYPES),
        default="Linear",
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_slice_thicknesses(
        self,
        option_string="--slice-thicknesses",
        nargs="+",
        type=float,
        help="Manually specify slice thicknesses of input image stacks. "
        "If not provided, the slice thicknesses of each acquired stack "
        "is assumed to be the image spacing in through-plane direction.",
        default=None,
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_viewer(
        self,
        option_string="--viewer",
        type=str,
        help="Viewer to be used for visualizations during verbose output "
        "(%s)." % ", ".join(VIEWER_OPTIONS),
        default="itksnap",
        required=False,
    ):
        self._add_argument(dict(locals()))

    def add_v2v_method(
        self,
        option_string="--v2v-method",
        type=str,
        help="Registration method used for first rigid volume-to-volume "
        "registration step "
        "(%s)." % ", ".join(V2V_METHOD_OPTIONS),
        default="FLIRT",
        required=False,
    ):
        self._add_argument(dict(locals()))

    ##
    # Parse the provided configuration file
    #
    # Additional arguments provided in the commandline will be preferred. E.g.
    # 'script.py --config config.json --dir-output path-to-output-dir' will set
    # dir-output to path-to-output-dir regardless the setting in config.json.
    # \date       2018-01-16 15:56:38+0000
    #
    # \param      self  The object
    # \post       sys.argv extended by the arguments provided in the config
    #             file
    #
    def _parse_config_file(self):

        # Read path to config file
        path_to_config_file = sys.argv[sys.argv.index(self._config_arg) + 1]

        # Read config file and insert all config entries into sys.argv (read by
        # argparse later)
        dic = ph.read_dictionary_from_json(path_to_config_file)

        # Insert all config entries into sys.argv
        for k, v in six.iteritems(dic):

            # ignore log info in config files
            if k in ["version", "user", "date"]:
                continue

            # A 'None' entry should be ignored
            if v is None:
                continue

            # Insert values as string right at the beginning of arguments
            # Rationale: Later options, outside of the config file, will
            # overwrite the config values
            if type(v) is list:
                for vi in reversed(v):
                    sys.argv.insert(1, str(vi))
            # 'store_true' values are converted to True/False; ignore the value
            # of this key as the existence of the key indicates 'True'
            elif type(v) is bool:
                if v == True:
                    sys.argv.insert(1, "--%s" % k)
                continue
            else:
                sys.argv.insert(1, str(v))

            sys.argv.insert(1, "--%s" % k)

    ##
    # Adds an argument to argument parser.
    #
    # Rationale: Make interface as generic as possible so that function call
    # works regardless the name of the desired option
    # \date       2017-08-06 21:54:51+0100
    #
    # \param      self     The object
    # \param      allvars  all variables set at respective function call as
    #                      dictionary
    #
    def _add_argument(self, allvars):

        # Skip variable 'self'
        allvars.pop('self')

        # Get name of argument to add
        option_string = allvars.pop('option_string')

        # Build dictionary for additional, optional parameters
        kwargs = {}
        for key, value in six.iteritems(allvars):
            kwargs[key] = value

        # Add information on default value in case provided
        if 'default' in kwargs.keys():

            if type(kwargs['default']) == list:
                txt = " ".join([str(i) for i in kwargs['default']])
            else:
                txt = str(kwargs['default'])
            txt_default = " [default: %s]" % txt

            # Case where 'required' key is given:
            if 'required' in kwargs.keys():

                # Only add information in case argument is not mandatory to
                # parse
                if kwargs['default'] is not None and not kwargs['required']:
                    kwargs['help'] += txt_default

            # Case where no such field was provided
            else:
                if kwargs['default'] is not None:
                    kwargs['help'] += txt_default

        # Add argument with its options
        self._parser.add_argument(option_string, **kwargs)
