## \file SegmentationPropagation.py
#  \brief 
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2017


## Import libraries
import sys
import itk
import SimpleITK as sitk
import numpy as np

## Import modules
import base.Stack as st
import utilities.SimpleITKHelper as sitkh
import utilities.PythonHelper as ph
import registration.RegistrationSimpleITK as regsitk
import registration.RegistrationITK as regitk
import registration.NiftyReg as regniftyreg

##
# Class implementing the segmentation propagation from one image to another
# \date       2017-05-10 23:48:08+0100
#
class SegmentationPropagation(object):

    ## Constructor
    def __init__(self, stack, template, registration_method=None, dilation_radius=0, dilation_kernel="Ball"):

        self._stack = stack
        self._template = template

        self._registration_method = registration_method

        self._dilation_radius = dilation_radius
        self._dilation_kernel = dilation_kernel

        self._get_registration_method = {
            "SimpleITK"     : regsitk,
            "ITK"           : regitk,
            "NiftyReg"      : regitk,
        }

        self._stack_sitk = sitk.Image(stack.sitk)
        self._stack_sitk_mask = None

        self._registration_transform_sitk = None

    def get_segmented_stack(self):
        
        ## Create new Stack instance
        stack_aligned_masked = st.Stack.from_sitk_image(self._stack_sitk, self._stack.get_filename(), self._stack_sitk_mask)

        return stack_aligned_masked


    def get_registration_transform_sitk(self):
        return self._registration_transform_sitk


    def run_segmentation_propagation(self, use_fixed_mask=True):

        ## Register stack to template
        if self._registration_method is not None:
            self._registration_method.set_fixed(self._template)
            self._registration_method.set_moving(self._stack)
            self._registration_method.use_fixed_mask(use_fixed_mask)
            self._registration_method.run_registration()

            self._registration_transform_sitk = self._registration_method.get_registration_transform_sitk()
            self._registration_transform_sitk = eval("sitk." + self._registration_transform_sitk.GetName() + "(self._registration_transform_sitk.GetInverse())")

            self._stack_sitk = sitkh.get_transformed_sitk_image(self._stack_sitk, self._registration_transform_sitk)

        ## Propagate mask
        self._stack_sitk_mask = sitk.Resample(self._template.sitk_mask, self._stack_sitk, sitk.Euler3DTransform(), sitk.sitkNearestNeighbor, 0, self._template.sitk_mask.GetPixelIDValue())

        ## Dilate mask
        if self._dilation_radius > 0:
            dilater = sitk.BinaryDilateImageFilter()
            dilater.SetKernelType(eval("sitk.sitk" + self._dilation_kernel))
            dilater.SetKernelRadius(self._dilation_radius)
            self._stack_sitk_mask = dilater.Execute(self._stack_sitk_mask)

