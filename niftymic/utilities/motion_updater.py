##
# \file motion_updater.py
# \brief      Class to apply stack and individual slice motion transformations
#             from a 'motion_correction' directory.
#
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2018
#

import os
import re
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh

import niftymic.base.stack as st
import niftymic.base.exceptions as exceptions


##
# Class to update motion correction of both stacks and slices given a directory
# that contains the respective transformation files.
#
# Provided motion-correction directory must contain "*.tfm" files in the
# following format:
#
# -# filenameA.tfm: Transformation to be applied to stack with filename
#    'filenameA'
# -# filenameA_slice[0-9]+.tfm: Transformations to be applied to individual
#    slices of stack with filename 'filenameA'. If a slice transformation file
#    is not provided, the respective slice will be deleted from the stack
# \date       2018-11-11 16:21:00+0000
#
class MotionUpdater(object):

    ##
    # { constructor_description }
    # \date       2018-11-11 16:26:58+0000
    #
    # \param      self                   The object
    # \param      stacks                 Stacks as list of Stack objects
    # \param      dir_motion_correction  Path to motion-correction files
    #                                    [*.tfm]
    # \param      prefix_slice           Prefix of slices to indicate slice
    #                                    transformations. E.g. "_slice" refers
    #                                    to filenameA_slice[0-9]+.tfm files as
    #                                    slice transformations to stack
    #                                    "filenameA"
    #
    def __init__(
        self,
        stacks,
        dir_motion_correction,
        prefix_slice="_slice",
    ):

        self._stacks = [st.Stack.from_stack(s) for s in stacks]
        self._dir_motion_correction = dir_motion_correction
        self._prefix_slice = prefix_slice

    def run(self):
        if not ph.directory_exists(self._dir_motion_correction):
            raise exceptions.DirectoryNotExistent(
                self._dir_motion_correction)
        abs_path_to_directory = os.path.abspath(
            self._dir_motion_correction)

        for i in range(len(self._stacks)):
            stack_name = self._stacks[i].get_filename()

            # update stack position
            path_to_stack_transform = os.path.join(
                abs_path_to_directory, "%s.tfm" % stack_name)
            if ph.file_exists(path_to_stack_transform):
                transform_stack_sitk = sitkh.read_transform_sitk(
                    path_to_stack_transform)
                transform_stack_sitk_inv = sitkh.read_transform_sitk(
                    path_to_stack_transform, inverse=True)
                self._stacks[i].update_motion_correction(
                    transform_stack_sitk)
                ph.print_info(
                    "Stack '%s': Stack position updated" % stack_name)
            else:
                transform_stack_sitk_inv = sitk.Euler3DTransform()

            # update slice positions
            pattern_trafo_slices = stack_name + self._prefix_slice + \
                "([0-9]+)[.]tfm"
            p = re.compile(pattern_trafo_slices)
            dic_slice_transforms = {
                int(p.match(f).group(1)): os.path.join(
                    abs_path_to_directory, p.match(f).group(0))
                for f in os.listdir(abs_path_to_directory) if p.match(f)
            }
            slices = self._stacks[i].get_slices()
            for i_slice in range(self._stacks[i].get_number_of_slices()):
                if i_slice in dic_slice_transforms.keys():
                    transform_slice_sitk = sitkh.read_transform_sitk(
                        dic_slice_transforms[i_slice])
                    transform_slice_sitk = \
                        sitkh.get_composite_sitk_affine_transform(
                            transform_slice_sitk, transform_stack_sitk_inv)
                    slices[i_slice].update_motion_correction(
                        transform_slice_sitk)

                    # # ------------------------- HACK -------------------------
                    # # 18 Jan 2019
                    # # HACK to use results of a previous version where image
                    # # slices were still exported
                    # # (Bug was that after stack intensity correction, the
                    # # previous v2v-reg was not passed on to the final 
                    # # registration transform):
                    # import niftymic.base.slice as sl
                    # # m = "_mask"
                    # m = "_BiliaryTree"
                    # path_to_slice = re.sub(
                    #     ".tfm", ".nii.gz", dic_slice_transforms[i_slice])
                    # path_to_slice_mask = re.sub(
                    #     ".tfm", "%s.nii.gz" % m, dic_slice_transforms[i_slice])
                    # slice_sitk = sitk.ReadImage(path_to_slice)
                    # slice_sitk_mask = sitk.ReadImage(path_to_slice_mask)
                    # hack = sl.Slice.from_sitk_image(
                    #     slice_sitk=slice_sitk,
                    #     # slice_sitk=slice_sitk_mask,  # mask for Mask-SRR!
                    #     slice_sitk_mask=slice_sitk_mask,
                    #     slice_number=slices[i_slice].get_slice_number(),
                    #     slice_thickness=slices[i_slice].get_slice_thickness(),
                    # )
                    # self._stacks[i]._slices[i_slice] = hack
                    # # --------------------------------------------------------

                else:
                    self._stacks[i].delete_slice(slices[i_slice])

            # print update information
            ph.print_info(
                "Stack '%s': Slice positions updated "
                "(%d/%d slices deleted)" % (
                    stack_name,
                    len(self._stacks[i].get_deleted_slice_numbers()),
                    self._stacks[i].sitk.GetSize()[-1],
                )
            )

            # delete entire stack if all slices were rejected
            if self._stacks[i].get_number_of_slices() == 0:
                ph.print_info(
                    "Stack '%s' removed as all slices were deleted" %
                    stack_name)
                self._stacks[i] = None

        # only return maintained stacks
        self._stacks = [s for s in self._stacks if s is not None]

        if len(self._stacks) == 0:
            raise RuntimeError(
                "All stacks removed. "
                "Did you check that the correct motion-correction directory "
                "was provided?")

    def get_data(self):
        return self._stacks
