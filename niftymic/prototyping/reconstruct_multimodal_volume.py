##
# \file reconstruct_multimodal_volume.py
# \brief      Script to reconstruct an isotropic, high-resolution volume from
#             multiple stacks of low-resolution 2D slices including
#             motion-correction.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       March 2017
#

# Import libraries
import os
import numpy as np
import SimpleITK as sitk

import niftymic.base.data_reader as dr
import niftymic.base.stack as st
import niftymic.reconstruction.primal_dual_solver as pd
import niftymic.reconstruction.scattered_data_approximation as \
    sda
import niftymic.reconstruction.tikhonov_solver as tk
import niftymic.registration.flirt as regflirt
import niftymic.registration.simple_itk_registration as regsitk
import niftymic.utilities.data_preprocessing as dp
import niftymic.utilities.segmentation_propagation as segprop
import niftymic.utilities.volumetric_reconstruction_pipeline as \
    pipeline
import niftymic.utilities.joint_image_mask_builder as imb
import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh
from niftymic.utilities.input_arparser import InputArgparser

import niftymic.prototyping.multi_modal_reconstruction as multirec


if __name__ == '__main__':
    # def main():

    time_start = ph.start_timing()

    # Set print options for numpy
    np.set_printoptions(precision=3)

    # Read input
    input_parser = InputArgparser(
        description="Volumetric MRI reconstruction framework to reconstruct "
        "an isotropic, high-resolution 3D volume from multiple "
        "motion-corrected (or static) stacks of low-resolution slices.",
    )
    input_parser.add_dir_input()
    input_parser.add_filenames()
    input_parser.add_image_selection()
    input_parser.add_dir_output(required=True)
    input_parser.add_suffix_mask(default="_mask")
    input_parser.add_target_stack_index(default=0)
    input_parser.add_extra_frame_target(default=10)
    input_parser.add_isotropic_resolution(default=None)
    input_parser.add_reconstruction_space(default=None)
    input_parser.add_minimizer(default="lsmr")
    input_parser.add_iter_max(default=10)
    input_parser.add_reconstruction_type(default="TK1L2")
    input_parser.add_data_loss(default="linear")
    input_parser.add_data_loss_scale(default=1)
    input_parser.add_alpha(
        default=0.02  # TK1L2
        # default=0.006  #TVL2, HuberL2
    )
    input_parser.add_rho(default=0.5)
    input_parser.add_tv_solver(default="PD")
    input_parser.add_pd_alg_type(default="ALG2")
    input_parser.add_iterations(default=15)
    input_parser.add_subfolder_comparison()
    input_parser.add_provide_comparison(default=0)
    input_parser.add_log_script_execution(default=1)
    input_parser.add_verbose(default=0)
    args = input_parser.parse_args()
    input_parser.print_arguments(args)

    # Write script execution call
    if args.log_script_execution:
        input_parser.write_performed_script_execution(
            os.path.abspath(__file__))

    # --------------------------------Read Data--------------------------------
    ph.print_title("Read Data")

    # Neither '--dir-input' nor '--filenames' was specified
    if args.filenames is not None and args.dir_input is not None:
        raise IOError(
            "Provide input by either '--dir-input' or '--filenames' "
            "but not both together")

    # '--dir-input' specified
    elif args.dir_input is not None:
        data_reader = dr.ImageSlicesDirectoryReader(
            path_to_directory=args.dir_input,
            suffix_mask=args.suffix_mask,
            image_selection=args.image_selection)

    # '--filenames' specified
    elif args.filenames is not None:
        data_reader = dr.MultipleImagesReader(
            args.filenames, suffix_mask=args.suffix_mask)

    else:
        raise IOError(
            "Provide input by either '--dir-input' or '--filenames'")

    if args.reconstruction_type not in ["TK1L2", "TVL2", "HuberL2"]:
        raise IOError("Reconstruction type unknown")

    data_reader.read_data()
    stacks = data_reader.get_stacks()

    # Reconstruction space is given isotropically resampled target stack
    if args.reconstruction_space is None:
        recon0 = \
            stacks[args.target_stack_index].get_isotropically_resampled_stack(
                resolution=args.isotropic_resolution,
                extra_frame=args.extra_frame_target)

    # Reconstruction space was provided by user
    else:
        recon0 = st.Stack.from_filename(args.reconstruction_space,
                                        extract_slices=False)

        # Change resolution for isotropic resolution if provided by user
        if args.isotropic_resolution is not None:
            recon0 = recon0.get_isotropically_resampled_stack(
                args.isotropic_resolution)

        # Use image information of selected target stack as recon0 serves
        # as initial value for reconstruction
        # recon0 = \
        #     stacks[args.target_stack_index].get_resampled_stack(recon0.sitk)

    stacks_dic = {
        0: [stacks[1], stacks[2], stacks[3]],
        1: [stacks[0], stacks[4]],
    }
    TE_dic = {
        0: 0.098,
        1: 0.066,
    }
    TR_dic = {
        0: 1.290,
        1: 0.887,
    }

    multi_modal_reconstruction = multirec.MultiModalReconstruction(
        stacks_dic=stacks_dic,
        TE_dic=TE_dic,
        TR_dic=TR_dic,
        reconstruction=recon0,
        iter_max=args.iter_max,
    )
    multi_modal_reconstruction.run()

    T2_sitk = multi_modal_reconstruction.get_T2_sitk()
    S0_sitk = multi_modal_reconstruction.get_S0_sitk()

    T1 = sitk.GetArrayFromImage(T2_sitk)
    T2 = sitk.GetArrayFromImage(T2_sitk)

    tmp = [recon0.sitk]
    label = ["recon0"]

    tmp.append(T2_sitk)
    label.append("T2")

    for m in stacks_dic.keys():
        TE_m = TE_dic[m]
        TR_m = TR_dic[m]
        S0_m = sitk.GetArrayFromImage(S0_sitk[m])
        tmp.append(S0_sitk[m])
        label.append("S0_%d" % m)

    for m in stacks_dic.keys():
        TE_m = TE_dic[m]
        TR_m = TR_dic[m]
        S0_m = sitk.GetArrayFromImage(S0_sitk[m])
        x_m = multirec.MRILaw.f(S0_m, TR_m, TE_m, T1, T2)
        x_m_sitk = sitk.GetImageFromArray(x_m)
        x_m_sitk.CopyInformation(recon0.sitk)
        tmp.append(x_m_sitk)
        title = "TE%s_TR%s" % (TE_m, TR_m)
        title = title.replace(".", "p")
        label.append(title)

    sitkh.show_sitk_image(tmp, label=label, dir_output=args.dir_output)

# T2_sitk =

# return 0


# if __name__ == '__main__':
#     main()
