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

## Add directories to import modules
dir_src_root = "../"
sys.path.append( dir_src_root )

## Import modules
import utilities.SimpleITKHelper as sitkh

## This class implements the interface to the Brain Extraction Tool (BET)
#  TODO
class BrainStripping(object):

    ##-------------------------------------------------------------------------
    # \brief      Constructor
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
    def __init__(self, compute_brain_image=False, compute_brain_mask=True, compute_skull_image=False, dir_tmp = "/tmp/BrainExtractionTool/", bet_options=""):

        self._compute_brain_image = compute_brain_image
        self._compute_brain_mask = compute_brain_mask
        self._compute_skull_image = compute_skull_image
        self._dir_tmp = dir_tmp
        self._bet_options = bet_options

        self._sitk = None
        self._sitk_brain_image = None
        self._sitk_brain_mask = None
        self._sitk_skull_image = None


    ##-------------------------------------------------------------------------
    # \brief      Initialize brain stripping class based on image to be read
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
    def from_filename(cls, dir_input, filename, compute_brain_image=False, compute_brain_mask=True, compute_skull_image=False, dir_tmp = "/tmp/BrainExtractionTool/"):

        self = cls(compute_brain_image=compute_brain_image, compute_brain_mask=compute_brain_mask, compute_skull_image=compute_skull_image, dir_tmp=dir_tmp)
        self._sitk = sitk.ReadImage(dir_input + filename + ".nii.gz", sitk.sitkFloat64)

        return self


    ##-------------------------------------------------------------------------
    # \brief      Initialize brain stripping class based on given sitk.Image
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
    def from_sitk_image(cls, sitk_image, compute_brain_image=False, compute_brain_mask=True, compute_skull_image=False, dir_tmp = "/tmp/BrainExtractionTool/"):

        self = cls(compute_brain_image=compute_brain_image, compute_brain_mask=compute_brain_mask, compute_skull_image=compute_skull_image, dir_tmp=dir_tmp)
        self._sitk = sitk.Image(sitk_image)

        return self

    ##-------------------------------------------------------------------------
    # \brief      Sets the sitk image for brain stripping
    # \date       2016-10-12 15:46:20+0100
    #
    # \param      self        The object
    # \param      sitk_image  The sitk image as sitk.Image object
    #
    #
    def set_input_image_sitk(self, sitk_image):
        self._sitk = sitk.Image(sitk_image)


    ##-------------------------------------------------------------------------
    # \brief      Set flag of whether or not to compute the brain image
    # \date       2016-10-12 12:35:46+0100
    #
    # \param      self                 The object
    # \param      compute_brain_image  Boolean flag
    #
    def compute_brain_image(self, compute_brain_image):
        self._compute_brain_image = compute_brain_image


    ##-------------------------------------------------------------------------
    # \brief      Set flag of whether or not to compute the brain image mask
    # \date       2016-10-12 12:36:46+0100
    #
    # \param      self                The object
    # \param      compute_brain_mask  Boolean flag
    #
    def compute_brain_mask(self, compute_brain_mask):
        self._compute_brain_mask = compute_brain_mask


    ##-------------------------------------------------------------------------
    # \brief      Set flag of whether or not to compute the skull mask
    # \date       2016-10-12 12:37:06+0100
    #
    # \param      self                The object
    # \param      compute_skull_image  Boolean flag
    #
    def compute_skull_image(self, compute_skull_image):
        self._compute_skull_image = compute_skull_image


    ##-------------------------------------------------------------------------
    # \brief      Set Brain Extraction Tool specific options
    # \date       2016-10-12 14:38:38+0100
    #
    # \param      self         The object
    # \param      bet_options  The bet options, string
    #
    def set_bet_options(self, bet_options):
        self._bet_options = bet_options


    ##-------------------------------------------------------------------------
    # \brief      Gets the input image
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


    ##-------------------------------------------------------------------------
    # \brief      Get computed brain image
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


    ##-------------------------------------------------------------------------
    # \brief      Get computed brain image mask
    # \date       2016-10-12 14:33:53+0100
    #
    # \param      self  The object
    #
    # \return     The brain mask as sitk.Image object
    #
    def get_brain_mask_sitk(self):
        if self._sitk_brain_mask is None:
            raise ValueError("Brain mask was not asked for. Set option '-m' and run again.")

        return self._sitk_brain_mask


    ##-------------------------------------------------------------------------
    # \brief      Get computed skull image mask
    # \date       2016-10-12 14:33:53+0100
    #
    # \param      self  The object
    #
    # \return     The skull mask image as sitk object.
    #
    def get_skull_image_sitk(self):
        if self._sitk_skull_image is None:
            raise ValueError("Skull mask was not asked for. Set option '-s' and run again.")

        return self._sitk_skull_image


    ##-------------------------------------------------------------------------
    # \brief      Run Brain Extraction Tool given the chosen set of parameters
    # \date       2016-10-12 14:59:01+0100
    #
    # \param      self  The object
    #
    def run_stripping(self):
        self._run_bet_for_brain_stripping()


    ##-------------------------------------------------------------------------
    # \brief      Run Brain Extraction Tool
    # \date       2016-10-12 14:59:24+0100
    #
    # \param      self  The object
    # \post       self._sitk* are filled with respective images
    #
    def _run_bet_for_brain_stripping(self):

        filename_out = "image"

        os.system("mkdir -p " + self._dir_tmp)
        os.system("rm -rf " + self._dir_tmp + "*")

        sitk.WriteImage(self._sitk, self._dir_tmp + filename_out + ".nii.gz")

        cmd  = "bet "
        cmd += self._dir_tmp + filename_out + ".nii.gz "
        cmd += self._dir_tmp + filename_out + "_bet.nii.gz "

        if not self._compute_brain_image:
            cmd += "-n "

        if self._compute_brain_mask:
            cmd += "-m "

        if self._compute_skull_image:
            cmd += "-s "

        cmd += self._bet_options + " "

        print(cmd)
        os.system(cmd)

        if self._compute_brain_image:
            self._sitk_brain_image = sitk.ReadImage(self._dir_tmp + filename_out + "_bet.nii.gz", sitk.sitkFloat64)

        if self._compute_brain_mask:
            self._sitk_brain_mask = sitk.ReadImage(self._dir_tmp + filename_out + "_bet_mask.nii.gz", sitk.sitkUInt8)

        if self._compute_skull_image:
            self._sitk_skull_image = sitk.ReadImage(self._dir_tmp + filename_out + "_bet_skull.nii.gz")
        
