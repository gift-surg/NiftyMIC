##
# \file ImageSimilarity.py
# \brief      Class containing functions to correct for intensities
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2016
# 
# \remark needs to be very flexible! For the moment I stick to the file 


## Import libraries 
import SimpleITK as sitk
import numpy as np
import sys
import os
from skimage.measure import compare_ssim as ssim

## Import modules
import utilities.SimpleITKHelper as sitkh
import utilities.PythonHelper as ph
import base.Stack as st


class ImageSimilarity(object):

    def __init__(self, reference=None, stacks=None):

        ## Number of input stacks
        self._N_stacks = len(stacks)
        
        ## Test that stacks and reference are in the same image space
        try:
            for i in range(0, self._N_stacks):
                stacks[i].sitk - reference.sitk
            
        except:
            raise ValueError("Reference and stack are not in the same space")

        ## Input stacks for comparison with reference
        self._stacks = [None]*self._N_stacks
        self._stacks_nda = [None]*self._N_stacks
        for i in range(0, self._N_stacks):
            self._stacks[i] = st.Stack.from_stack(stacks[i])
            self._stacks_nda[i] = sitk.GetArrayFromImage(self._stacks[i].sitk)
        
        ## Reference stack
        self._reference = st.Stack.from_stack(reference)
        self._reference_nda = sitk.GetArrayFromImage(self._reference.sitk)
        self._reference_mask_nda = sitk.GetArrayFromImage(self._reference.sitk_mask)

        ## Similarity measures
        self._ssim = [None]*self._N_stacks
        self._psnr = [None]*self._N_stacks
        self._nmi = [None]*self._N_stacks
        self._mse = [None]*self._N_stacks


    def compute_structural_similarity(self, use_reference_mask=True):

        reference_nda = np.array(self._reference_nda)    
        if use_reference_mask:
            reference_nda *= self._reference_mask_nda

        for i in range(0, self._N_stacks):
            stack_nda = np.array(self._stacks_nda[i])
            if use_reference_mask:
                stack_nda *= self._reference_mask_nda

            ## Compute Structural Similarity Measure
            self._ssim[i] = ssim(stack_nda, reference_nda)
            
        return np.array(self._ssim)


    def compute_mean_squared_error(self, use_reference_mask=True):

        reference_nda = np.array(self._reference_nda)    
        if use_reference_mask:
            reference_nda *= self._reference_mask_nda
            N = self._reference_mask_nda.sum().astype('float')
        else:
            N = np.array(self._reference.sitk.GetSize()).prod().astype('float')

        for i in range(0, self._N_stacks):
            stack_nda = np.array(self._stacks_nda[i])
            if use_reference_mask:
                stack_nda *= self._reference_mask_nda

            self._mse[i] = ((stack_nda - reference_nda)**2).sum()/N

        return np.array(self._mse)
        

    def compute_peak_signal_to_noise_ratio(self, use_reference_mask=True):

        self.compute_mean_squared_error(use_reference_mask)

        for i in range(0, self._N_stacks):
            stack_nda = np.array(self._stacks_nda[i])
            if use_reference_mask:
                stack_nda *= self._reference_mask_nda

            self._psnr[i] = 10*np.log10(np.max(stack_nda)**2/self._mse[i])

        return np.array(self._psnr)