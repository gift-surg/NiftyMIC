## \file StackAverage.py
#  \brief Implementation of stack averaging on isotropic grid
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date April 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import itk
import SimpleITK as sitk
import numpy as np
import time                     

## Import modules from src-folder
import utilities.SimpleITKHelper as sitkh
import base.Stack as st


## Class implementing averaging of stacks as a crude approximation of isotropic 
#  volume
class StackAverage:

    ## Constructor
    #  \param[in] stack_manager instance of StackManager containing all stacks and additional information
    #  \param[in] HR_volume Stack object containing the current estimate of the HR volume (required for defining HR space)
    def __init__(self, stack_manager, HR_volume=None):

        ## Initialize variables
        self._stack_manager = stack_manager
        self._stacks = stack_manager.get_stacks()
        self._N_stacks = stack_manager.get_number_of_stacks()
        
        self._averaged_volume_name = "average"

        self._mask_volume_voxels = True

        ## If HR volume is not given then resample first stack to get reference space
        if HR_volume is None:
            self._averaged_volume = self._get_isotropically_resampled_stack(self._stacks[0])
        else:
            self._averaged_volume = HR_volume

        self._transformations = [sitk.Euler3DTransform()]*self._N_stacks



    ## Set of averaged volume
    #  \param[in] name string containing the chosen name for the stack
    def set_averaged_volume_name(self, name):
        self._averaged_volume_name = name


    ## Set transformations applied on each stack prior averaging
    #  \remark: Only used for in-plane registration and can be deleted at some point
    #  \param[in] transformations as sitk objects
    def set_stack_transformations(self, transformations):
        self._transformations = transformations


    ## Get averaged, isotropic volume of given stacks 
    #  \return current estimate of HR volume, instance of Stack
    def get_averaged_volume(self):
        return self._averaged_volume


    ## Specify whether non masked voxels shall be set to zero
    def set_mask_volume_voxels(self, mask_volume_voxels):
        self._mask_volume_voxels = mask_volume_voxels


    ## Compute average of all given stacks over specified target stack
    def run_averaging(self):
        default_pixel_value = 0.0

        ## Define helpers to obtain averaged stack
        shape = sitk.GetArrayFromImage(self._averaged_volume.sitk).shape
        array = np.zeros(shape)
        array_mask = np.zeros(shape)
        ind = np.zeros(shape)


        ## Average over domain specified by the joint mask ("union mask")
        for i in range(0,self._N_stacks):
            ## Resample warped stacks
            stack_sitk =  sitk.Resample(
                self._stacks[i].sitk,
                self._averaged_volume.sitk, 
                self._transformations[i], 
                sitk.sitkLinear, 
                default_pixel_value,
                self._averaged_volume.sitk.GetPixelIDValue())

            ## Resample warped stack masks
            stack_sitk_mask =  sitk.Resample(
                self._stacks[i].sitk_mask,
                self._averaged_volume.sitk, 
                self._transformations[i], 
                sitk.sitkNearestNeighbor, 
                default_pixel_value,
                self._stacks[i].sitk_mask.GetPixelIDValue())

            ## Get arrays of resampled warped stack and mask
            array_tmp = sitk.GetArrayFromImage(stack_sitk)
            array_mask_tmp = sitk.GetArrayFromImage(stack_sitk_mask)

            ## Sum intensities of stack and mask
            array += array_tmp
            array_mask += array_mask_tmp

            ## Store indices of voxels with non-zero contribution
            ind[np.nonzero(array_tmp)] += 1

        ## Average over the amount of non-zero contributions of the stacks at each index
        ind[ind==0] = 1                 # exclude division by zero
        array = np.divide(array,ind.astype(float))    # elemenwise division

        ## Create (joint) binary mask. Mask represents union of all masks
        array_mask[array_mask>0] = 1

        ## Set pixels of the image not specified by the mask to zero
        if self._mask_volume_voxels:
            array[array_mask==0] = 0

        ## Update HR volume (sitk image)
        helper = sitk.GetImageFromArray(array)
        helper.CopyInformation(self._averaged_volume.sitk)
        self._averaged_volume.sitk = helper
        self._averaged_volume.itk = sitkh.get_itk_from_sitk_image(helper)

        ## Update HR volume (sitk image mask)
        # helper = sitk.GetImageFromArray(array_mask)
        # helper.CopyInformation(self._averaged_volume.sitk_mask)
        # self._averaged_volume.sitk_mask = helper


    ## Resample stack to isotropic grid
    #  The image and its mask get resampled to isotropic grid 
    #  (in-plane resolution also in through-plane direction)
    #  \param[in] target_stack Stack being resampled
    #  \return Isotropically resampled Stack
    def _get_isotropically_resampled_stack(self, target_stack):
        
        ## Read original spacing (voxel dimension) and size of target stack:
        spacing = np.array(target_stack.sitk.GetSpacing())
        size = np.array(target_stack.sitk.GetSize())

        ## Update information according to isotropic resolution
        size[2] = np.round(spacing[2]/spacing[0]*size[2])
        spacing[2] = spacing[0]

        ## Resample image and its mask to isotropic grid
        default_pixel_value = 0.0

        HR_volume_sitk =  sitk.Resample(
            target_stack.sitk, 
            size, 
            sitk.Euler3DTransform(), 
            sitk.sitkNearestNeighbor, 
            target_stack.sitk.GetOrigin(), 
            spacing,
            target_stack.sitk.GetDirection(),
            default_pixel_value,
            target_stack.sitk.GetPixelIDValue())

        HR_volume_sitk_mask =  sitk.Resample(
            target_stack.sitk_mask, 
            size, 
            sitk.Euler3DTransform(), 
            sitk.sitkNearestNeighbor, 
            target_stack.sitk.GetOrigin(), 
            spacing,
            target_stack.sitk.GetDirection(),
            default_pixel_value,
            target_stack.sitk_mask.GetPixelIDValue())

        ## Create Stack instance of HR_volume
        HR_volume = st.Stack.from_sitk_image(HR_volume_sitk, self._averaged_volume_name, HR_volume_sitk_mask)

        return HR_volume
