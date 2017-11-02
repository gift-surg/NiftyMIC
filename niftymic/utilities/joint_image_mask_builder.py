##
# \file joint_image_mask_builder.py
# \brief      Build common mask from multiple, individual ones
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       November 2017
#

import SimpleITK as sitk
import numpy as np

import pysitk.simple_itk_helper as sitkh
import niftymic.base.stack as st


class JointImageMaskBuilder(object):

    def __init__(self,
                 stacks,
                 target,
                 dilation_radius=1,
                 dilation_kernel="Ball",
                 max_distance=50):

        self._stacks = stacks
        self._target = target
        self._dilation_radius = dilation_radius
        self._dilation_kernel = dilation_kernel
        self._max_distance = max_distance

        self._joint_image_mask = None

    def run(self):

        recon_space = self._target.get_isotropically_resampled_stack(
            extra_frame=self._max_distance,
        )
        mask_sitk = 0 * recon_space.sitk_mask
        dim = mask_sitk.GetDimension()

        for stack in self._stacks:
            stack_mask_sitk = sitk.Resample(
                stack.sitk_mask,
                mask_sitk,
                eval("sitk.Euler%dDTransform()" % dim),
                sitk.sitkNearestNeighbor,
                0,
                mask_sitk.GetPixelIDValue())
            mask_sitk += stack_mask_sitk

        thresholder = sitk.BinaryThresholdImageFilter()
        mask_sitk = thresholder.Execute(mask_sitk, 0, 0.5, 0, 1)

        if self._dilation_radius > 0:
            dilater = sitk.BinaryDilateImageFilter()
            dilater.SetKernelType(eval("sitk.sitk" + self._dilation_kernel))
            dilater.SetKernelRadius(self._dilation_radius)
            mask_sitk = dilater.Execute(mask_sitk)

        self._joint_image_mask = st.Stack.from_sitk_image(
            image_sitk=recon_space.sitk,
            image_sitk_mask=mask_sitk,
            filename=self._target.get_filename(),
        )

    def get_stack(self):
        return st.Stack.from_stack(self._joint_image_mask)
