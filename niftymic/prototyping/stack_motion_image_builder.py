##
# \file stack_motion_image_builder.py
# \brief      Class to build image that shows estimated intra-stack motion,
#             i.e. motion for each individual slice
#
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2018
# \todo Currently obsolete


import os
import re
import numpy as np
import SimpleITK as sitk

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh
import simplereg.utilities

import niftymic.base.stack as st
import niftymic.base.exceptions as exceptions


##
# Front-end to create stack motion image objects step by step.
# \date       2018-11-11 17:06:15+0000
#
class StackMotionImageBuilder(object):

    def __init__(self, stack=None, stack_ref=None):
        self._stack = stack
        self._stack_ref = stack_ref

    def set_stack(self, stack):
        self._stack = stack

    def set_stack_ref(self, stack_ref):
        self._stack_ref = stack_ref

    def _check_inputs(self):
        if self._stack_ref is None:
            raise ValueError("Reference stack needs to be provided")

        if self._stack_ref is not None:
            if self._stack.sitk.GetSize() != self._stack_ref.sitk.GetSize():
                raise ValueError(
                    "Provided stacks must have identical data array sizes")

    def get_voxel_displacements_image(self):
        self._check_inputs()

        shape = self._stack.sitk.GetSize()[::-1]
        nda = np.ones(shape) * np.inf

        slices_ref = self._stack_ref.get_slices()
        slices_ref_dic = {s.get_slice_number(): s for s in slices_ref}

        # Compute mean displacements for all (non-discarded) slices
        for slice in self._stack.get_slices():

            i = slice.get_slice_number()
            nda[i, :, :] = \
                simplereg.utilities.get_voxel_displacements(
                    slice.sitk, slices_ref_dic[i].sitk
            )

        return self._get_stack_from_nda(nda)

    ##
    # Gets the absolute motion image, i.e. l1-norm of rigid transform
    # parameters
    # \date       2018-11-14 13:09:47+0000
    #
    # \param      self  The object
    #
    # \return     The absolute motion image as stack object.
    #
    def get_absolute_motion_image(self):
        shape = self._stack.sitk.GetSize()[::-1]
        nda = np.ones(shape) * np.inf

        slices = self._stack.get_slices()
        for slice in slices:
            k = slice.get_slice_number()

            parameters_nda = self._get_rigid_parameters(slice)

            # Convert rotation parameters to degrees
            parameters_nda[0:3] *= 180. / np.pi

            # Compute l1-norm of motion parameters
            nda[k, :, :] = np.sum(np.abs(parameters_nda))

        return self._get_stack_from_nda(nda)

    ##
    # Gets the motion parameter image, i.e. multi-component image with all
    # rigid transform parameters
    # \date       2018-11-14 13:08:01+0000
    #
    # \param      self  The object
    #
    # \return     The motion image as stack object.
    #
    def get_motion_parameter_image(self):

        # Assume rigid 3D motion, i.e. 6 degrees of freedom
        shape = self._stack.sitk.GetSize()[::-1] + (6,)
        nda = np.ones(shape, np.float32) * np.inf

        slices = self._stack.get_slices()
        for slice in slices:
            k = slice.get_slice_number()

            parameters_nda = self._get_rigid_parameters(slice)

            # Convert rotation parameters to degrees
            parameters_nda[0:3] *= 180. / np.pi

            nda[k, :, :] = parameters_nda

        return self._get_stack_from_nda(nda)

    @staticmethod
    def _get_rigid_parameters(slice):
        transform_sitk = slice.get_motion_correction_transform()
        euler_sitk = simplereg.utilities.extract_rigid_from_affine(
            transform_sitk)
        parameters_nda = np.array(euler_sitk.GetParameters())

        return parameters_nda

    def _get_stack_from_nda(self, nda):
        image_sitk = sitk.GetImageFromArray(nda)
        image_sitk.SetOrigin(self._stack_ref.sitk.GetOrigin())
        image_sitk.SetSpacing(self._stack_ref.sitk.GetSpacing())
        image_sitk.SetDirection(self._stack_ref.sitk.GetDirection())

        stack = st.Stack.from_sitk_image(
            image_sitk=image_sitk,
            filename=self._stack.get_filename(),
            slice_thickness=self._stack_ref.get_slice_thickness(),
            image_sitk_mask=self._stack_ref.sitk_mask,
            extract_slices=False,
        )
        return stack
