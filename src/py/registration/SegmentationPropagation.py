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
import utilities.StackMaskMorphologicalOperations as stmorph
import registration.RegistrationSimpleITK as regsitk
import registration.RegistrationITK as regitk
import registration.NiftyReg as regniftyreg

##
# Class implementing the segmentation propagation from one image to another
# \date       2017-05-10 23:48:08+0100
#
class SegmentationPropagation(object):

    ## Constructor
    def __init__(self, stack=None, template=None, registration_method=None, use_template_mask=True, dilation_radius=0, dilation_kernel="Ball", use_dilation_in_plane_only=True):

        self._stack = stack
        self._template = template

        self._registration_method = registration_method

        self._dilation_radius = dilation_radius
        self._dilation_kernel = dilation_kernel
        self._use_dilation_in_plane_only = use_dilation_in_plane_only

        self._get_registration_method = {
            "SimpleITK"     : regsitk,
            "ITK"           : regitk,
            "NiftyReg"      : regitk,
        }

        self._stack_sitk = None
        self._stack_sitk_mask = None
        self._registration_transform_sitk = None
        self._use_template_mask = use_template_mask

    def set_stack(self, stack):
        self._stack = stack

    def get_stack(self):
        return self._stack

    def set_template(self, template):
        self._template = template

    def get_template(self):
        return self._template

    def set_dilation_radius(self, dilation_radius):
        self._dilation_radius = dilation_radius

    def get_dilation_radius(self):
        return self._dilation_radius

    def set_dilation_kernel(self, dilation_kernel):
        if dilation_kernel not in ['Ball', 'Box', 'Annulus', 'Cross']:
            raise ValueError("Dilation kernel must be 'Ball', 'Box', 'Annulus' or 'Cross'.")
        self._dilation_kernel = dilation_kernel

    def get_dilation_kernel(self):
        return self._dilation_kernel

    def get_segmented_stack(self):
        
        ## Create new Stack instance
        stack_aligned_masked = st.Stack.from_sitk_image(self._stack_sitk, self._stack.get_filename(), self._stack_sitk_mask)

        return stack_aligned_masked


    def get_registration_transform_sitk(self):
        return self._registration_transform_sitk


    def run_segmentation_propagation(self):

        if self._stack is None or self._template is None:
            raise ValueError("Specify stack and template first")

        self._stack_sitk = sitk.Image(self._stack.sitk)

        ## Register stack to template
        if self._registration_method is not None:
            self._registration_method.set_fixed(self._template)
            self._registration_method.set_moving(self._stack)
            self._registration_method.use_fixed_mask(self._use_template_mask)
            self._registration_method.run_registration()

            self._registration_transform_sitk = self._registration_method.get_registration_transform_sitk()
            self._registration_transform_sitk = eval("sitk." + self._registration_transform_sitk.GetName() + "(self._registration_transform_sitk.GetInverse())")

            self._stack_sitk = sitkh.get_transformed_sitk_image(self._stack_sitk, self._registration_transform_sitk)

        ## Propagate mask
        self._stack_sitk_mask = sitk.Resample(self._template.sitk_mask, self._stack_sitk, sitk.Euler3DTransform(), sitk.sitkNearestNeighbor, 0, self._template.sitk_mask.GetPixelIDValue())

        ## Dilate mask
        if self._dilation_radius > 0:

            stack_mask_morpher = stmorph.StackMaskMorphologicalOperations.from_sitk_mask(
                mask_sitk=self._stack_sitk_mask,
                dilation_radius=self._dilation_radius,
                dilation_kernel=self._dilation_kernel,
                use_dilation_in_plane_only=self._use_dilation_in_plane_only,
            )
            stack_mask_morpher.run_dilation()
            self._stack_sitk_mask = stack_mask_morpher.get_processed_mask_sitk()

            # sitkh.show_sitk_image(self._stack_sitk_mask)

