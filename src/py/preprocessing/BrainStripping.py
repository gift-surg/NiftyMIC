#!/usr/bin/python

## \file BrainStripping.py
#  \brief This class implements the interface to the Brain Extraction Tool 
#       (BET) to automatically segment the brain and/or the skull.
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Oct 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import itk
import SimpleITK as sitk
import numpy as np

## Import modules
import pythonhelper.SimpleITKHelper as sitkh
import pythonhelper.PythonHelper as ph

from definitions import DIR_TMP
from definitions import BET_EXE

## This class implements the interface to the Brain Extraction Tool (BET)
#  TODO
class BrainStripping(object):

    ##
    #       Constructor
    # \date       2016-10-12 12:43:38+0100
    #
    # \param      self                 The object
    # \param      compute_brain_image  Boolean flag for computing brain image
    # \param      compute_brain_mask   Boolean flag for computing brain image
    #                                  mask
    # \param      compute_skull_image   Boolean flag for computing skull mask
    # \param      dir_tmp              Directory where temporary results are
    #                                  written to, string
    #
    def __init__(self, compute_brain_image=False, compute_brain_mask=True, compute_skull_image=False, dir_tmp=os.path.join(DIR_TMP, "BrainExtractionTool"), bet_options=""):

        self._compute_brain_image = compute_brain_image
        self._compute_brain_mask = compute_brain_mask
        self._compute_skull_image = compute_skull_image
        self._dir_tmp = dir_tmp
        self._bet_options = bet_options

        self._sitk = None
        self._sitk_brain_image = None
        self._sitk_brain_mask = None
        self._sitk_skull_image = None


    ##
    #       Initialize brain stripping class based on image to be read
    # \date       2016-10-12 12:19:18+0100
    #
    # \param      cls                  The cls
    # \param      dir_input            The dir input
    # \param      filename             The filename
    # \param      compute_brain_image  Boolean flag for computing brain image
    # \param      compute_brain_mask   Boolean flag for computing brain image
    #                                  mask
    # \param      compute_skull_image   Boolean flag for computing skull mask
    # \param      dir_tmp              Directory where temporary results are
    #                                  written to, string
    #
    @classmethod
    def from_filename(cls, dir_input, filename, compute_brain_image=False, compute_brain_mask=True, compute_skull_image=False, dir_tmp=os.path.join(DIR_TMP, "BrainExtractionTool")):

        self = cls(compute_brain_image=compute_brain_image, compute_brain_mask=compute_brain_mask, compute_skull_image=compute_skull_image, dir_tmp=dir_tmp)
        self._sitk = sitk.ReadImage(dir_input + filename + ".nii.gz", sitk.sitkFloat64)

        return self


    ##
    #       Initialize brain stripping class based on given sitk.Image
    #             object
    # \date       2016-10-12 12:18:35+0100
    #
    # \param      cls                  The cls
    # \param      sitk_image           The sitk image
    # \param      compute_brain_image  Boolean flag for computing brain image
    # \param      compute_brain_mask   Boolean flag for computing brain image
    #                                  mask
    # \param      compute_skull_image   Boolean flag for computing skull mask
    # \param      dir_tmp              Directory where temporary results are
    #                                  written to, string
    #
    # \return     { description_of_the_return_value }
    #
    @classmethod
    def from_sitk_image(cls, sitk_image, compute_brain_image=False, compute_brain_mask=True, compute_skull_image=False, dir_tmp=os.path.join(DIR_TMP, "BrainExtractionTool")):

        self = cls(compute_brain_image=compute_brain_image, compute_brain_mask=compute_brain_mask, compute_skull_image=compute_skull_image, dir_tmp=dir_tmp)
        self._sitk = sitk.Image(sitk_image)

        return self

    ##
    #       Sets the sitk image for brain stripping
    # \date       2016-10-12 15:46:20+0100
    #
    # \param      self        The object
    # \param      sitk_image  The sitk image as sitk.Image object
    #
    #
    def set_input_image_sitk(self, sitk_image):
        self._sitk = sitk.Image(sitk_image)


    ##
    #       Set flag of whether or not to compute the brain image
    # \date       2016-10-12 12:35:46+0100
    #
    # \param      self                 The object
    # \param      compute_brain_image  Boolean flag
    #
    def compute_brain_image(self, compute_brain_image):
        self._compute_brain_image = compute_brain_image


    ##
    #       Set flag of whether or not to compute the brain image mask
    # \date       2016-10-12 12:36:46+0100
    #
    # \param      self                The object
    # \param      compute_brain_mask  Boolean flag
    #
    def compute_brain_mask(self, compute_brain_mask):
        self._compute_brain_mask = compute_brain_mask


    ##
    #       Set flag of whether or not to compute the skull mask
    # \date       2016-10-12 12:37:06+0100
    #
    # \param      self                The object
    # \param      compute_skull_image  Boolean flag
    #
    def compute_skull_image(self, compute_skull_image):
        self._compute_skull_image = compute_skull_image


    ##
    #       Set Brain Extraction Tool specific options
    # \date       2016-10-12 14:38:38+0100
    #
    # \param      self         The object
    # \param      bet_options  The bet options, string
    #
    def set_bet_options(self, bet_options):
        self._bet_options = bet_options


    ##
    #       Gets the input image
    # \date       2016-10-12 14:41:05+0100
    #
    # \param      self  The object
    #
    # \return     The input image as sitk.Image object
    #
    def get_input_image_sitk(self):
        if self._sitk is None:
            raise ValueError("Input image was not read yet.")

        return sitk.Image(self._sitk)


    ##
    #       Get computed brain image
    # \date       2016-10-12 14:33:53+0100
    #
    # \param      self  The object
    #
    # \return     The brain image as sitk object.
    #
    def get_brain_image_sitk(self):
        if self._sitk_brain_image is None:
            raise ValueError("Brain was not asked for. Do not set option '-n' and run again.")

        return self._sitk_brain_image


    ##
    #       Get computed brain image mask
    # \date       2016-10-12 14:33:53+0100
    #
    # \param      self  The object
    #
    # \return     The brain mask as sitk.Image object
    #
    def get_brain_mask_sitk(self, dilate_radius=0):
        if self._sitk_brain_mask is None:
            raise ValueError("Brain mask was not asked for. Set option '-m' and run again.")

        if dilate_radius > 0:
            ## Chose kernel
            kernel_sitk = sitk.sitkBall
            # kernel_sitk = sitk.sitkBox
            # kernel_sitk = sitk.sitkAnnulus
            # kernel_sitk = sitk.sitkCross

            ## Define dilate and erode image filter
            dilater = sitk.BinaryDilateImageFilter()
            dilater.SetKernelType(kernel_sitk)
            dilater.SetKernelRadius(dilate_radius)
            brain_mask_sitk = dilater.Execute(self._sitk_brain_mask)
        else:
            brain_mask_sitk = sitk.Image(self._sitk_brain_mask)

        return brain_mask_sitk


    ##
    # Get computed skull image mask
    # \date       2016-10-12 14:33:53+0100
    #
    # \param      self           The object
    # \param      dilate_radius  The dilate radius
    # \param      erode_radius   The erode radius
    # \param      kernel         The kernel in "Ball", "Box", "Annulus" or "Cross"
    #
    # \return     The skull mask image as sitk object.
    #
    def get_skull_mask_sitk(self, dilate_radius=10, erode_radius=0, kernel="Ball"):
        if self._sitk_skull_image is None:
            raise ValueError("Skull mask was not asked for. Set option '-s' and run again.")

        skull_mask_sitk = sitk.Image(self._sitk_skull_image)

        ## Skull mask from BET has values of either 0 or 100. Threshold to 0,1
        thresholder = sitk.BinaryThresholdImageFilter()
        thresholder.SetUpperThreshold(255)
        thresholder.SetLowerThreshold(1)
        skull_mask_sitk = thresholder.Execute(skull_mask_sitk)

        ## Translate kernel
        kernel_sitk = eval("sitk.sitk" + kernel)

        ## Define dilate and erode image filter
        if dilate_radius > 0:
            dilater = sitk.BinaryDilateImageFilter()
            dilater.SetKernelType(kernel_sitk)
            dilater.SetKernelRadius(dilate_radius)
            skull_mask_sitk = dilater.Execute(skull_mask_sitk)

        if erode_radius > 0:
            eroder = sitk.BinaryErodeImageFilter()
            eroder.SetKernelType(kernel_sitk)
            eroder.SetKernelRadius(erode_radius)
            skull_mask_sitk = eroder.Execute(skull_mask_sitk)

        return skull_mask_sitk


    ##
    #       Gets the mask around skull which covers also a bit of the
    #             brain. (It was used for the MS project)
    # \date       2016-11-06 22:54:28+0000
    #
    # \param      self           The object
    # \param      dilate_radius  The dilate radius
    # \param      erode_radius   The erode radius
    # \param      kernel         The kernel in "Ball", "Box", "Annulus" or "Cross"
    #
    # \return     The mask around skull.
    #
    def get_mask_around_skull(self, dilate_radius=10, erode_radius=0, kernel="Ball"):

        ## Translate kernel
        kernel_sitk = eval("sitk.sitk" + kernel)

        ## Define dilate and erode image filter
        dilater = sitk.BinaryDilateImageFilter()
        dilater.SetKernelType(kernel_sitk)
        dilater.SetKernelRadius(dilate_radius)

        eroder = sitk.BinaryErodeImageFilter()
        eroder.SetKernelType(kernel_sitk)
        eroder.SetKernelRadius(erode_radius)

        ## Get complement of brain mask
        mask_sitk = 1 - self._sitk_brain_mask
        
        shape = np.array(self._sitk_brain_mask.GetSize()[::-1])
        mask_nda = np.zeros((shape[0], shape[1], shape[2]))
        
        ## Go slice by slice
        for i in range(0, shape[0]):
            slice_mask_sitk = mask_sitk[:,:,i:i+1]

            ## Dilate mask of slice    
            if dilate_radius > 0:
                slice_mask_sitk = dilater.Execute(slice_mask_sitk)

            ## Erode mask of slice
            if erode_radius > 0:
                slice_mask_sitk = eroder.Execute(slice_mask_sitk)
            
            ## Fill data array information
            mask_nda[i,:,:] = sitk.GetArrayFromImage(slice_mask_sitk)

        ## Convert mask back to 3D image
        skull_mask_sitk = sitk.GetImageFromArray(mask_nda)
        skull_mask_sitk.CopyInformation(self._sitk_brain_mask)

        ## Debug:
        # sitkh.show_sitk_image(self._sitk, segmentation=skull_mask_sitk, title="stack_brain_mask")


        return skull_mask_sitk


    ##
    #       Run Brain Extraction Tool given the chosen set of parameters
    # \date       2016-10-12 14:59:01+0100
    #
    # \param      self  The object
    #
    def run_stripping(self):
        self._run_bet_for_brain_stripping()


    ##
    #       Run Brain Extraction Tool
    # \date       2016-10-12 14:59:24+0100
    #
    # \param      self  The object
    # \post       self._sitk* are filled with respective images
    #
    def _run_bet_for_brain_stripping(self):

        filename_out = "image"

        self._dir_tmp = ph.create_directory(self._dir_tmp, delete_files=True) 

        sitk.WriteImage(self._sitk, self._dir_tmp + filename_out + ".nii.gz")

        cmd  = BET_EXE + " "
        cmd += self._dir_tmp + filename_out + ".nii.gz "
        cmd += self._dir_tmp + filename_out + "_bet.nii.gz "

        if not self._compute_brain_image:
            cmd += "-n "

        if self._compute_brain_mask:
            cmd += "-m "

        if self._compute_skull_image:
            cmd += "-s "

        cmd += self._bet_options + " "

        ph.execute_command(cmd)

        if self._compute_brain_image:
            self._sitk_brain_image = sitk.ReadImage(self._dir_tmp + filename_out + "_bet.nii.gz", sitk.sitkFloat64)

        if self._compute_brain_mask:
            self._sitk_brain_mask = sitk.ReadImage(self._dir_tmp + filename_out + "_bet_mask.nii.gz", sitk.sitkUInt8)

        if self._compute_skull_image:
            self._sitk_skull_image = sitk.ReadImage(self._dir_tmp + filename_out + "_bet_skull.nii.gz")
        
