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

    def __init__(self, reference=None, stacks=None, use_reference_mask=True):

        ## Number of input stacks
        self._N_stacks = len(stacks)

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

        ##
        self._use_reference_mask = use_reference_mask

        ## Similarity measures
        self._ssim = [None]*self._N_stacks
        self._psnr = [None]*self._N_stacks
        self._nmi = [None]*self._N_stacks
        self._mse = [None]*self._N_stacks
        self._ncc = [None]*self._N_stacks


    def use_reference_mask(self, use_reference_mask):
        self._use_reference_mask = use_reference_mask


    def compute_structural_similarity(self):

        ## Test that all images are in the same physical space
        self._test_space(self._reference, self._stacks)

        ## Get data array information for reference
        reference_nda = np.array(self._reference_nda)    
        if self._use_reference_mask:
            reference_nda *= self._reference_mask_nda

        ## Get data array information for stacks
        for i in range(0, self._N_stacks):
            stack_nda = np.array(self._stacks_nda[i])
            if self._use_reference_mask:
                stack_nda *= self._reference_mask_nda

            ## Compute Structural Similarity Measure
            self._ssim[i] = ssim(stack_nda, reference_nda)
            
        return np.array(self._ssim)


    def compute_mean_squared_error(self):

        ## Test that all images are in the same physical space
        self._test_space(self._reference, self._stacks)

        ## Get data array information for reference
        reference_nda = np.array(self._reference_nda)    
        if self._use_reference_mask:
            reference_nda *= self._reference_mask_nda
            N = self._reference_mask_nda.sum().astype('float')
        else:
            N = np.array(self._reference.sitk.GetSize()).prod().astype('float')

        ## Get data array information for stacks
        for i in range(0, self._N_stacks):
            stack_nda = np.array(self._stacks_nda[i])
            if self._use_reference_mask:
                stack_nda *= self._reference_mask_nda

            ## Compute Sum of Squared Differences
            ssd = np.sum(np.square(stack_nda - reference_nda))
            
            ## Compute Mean Squared Error/Differences
            self._mse[i] = ssd/N

        return np.array(self._mse)
    

    def compute_peak_signal_to_noise_ratio(self):

        self.compute_mean_squared_error()

        for i in range(0, self._N_stacks):
            stack_nda = np.array(self._stacks_nda[i])
            if self._use_reference_mask:
                stack_nda *= self._reference_mask_nda

            self._psnr[i] = 10*np.log10(np.max(stack_nda)**2/self._mse[i])

        return np.array(self._psnr)


    def compute_normalized_cross_correlation(self):

        ## Test that all images are in the same physical space
        self._test_space(self._reference, self._stacks)

        ## Get data array information for reference
        reference_nda = np.array(self._reference_nda)    
        if self._use_reference_mask:
            reference_nda *= self._reference_mask_nda
            N = self._reference_mask_nda.sum().astype('float')
        else:
            N = np.array(self._reference.sitk.GetSize()).prod().astype('float')

        ## Get data array information for stacks
        for i in range(0, self._N_stacks):
            stack_nda = np.array(self._stacks_nda[i])
            if self._use_reference_mask:
                stack_nda *= self._reference_mask_nda

            ## Compute Normalized Cross Correlation
            mean_ref = np.sum(reference_nda)/N
            std_ref =  np.sqrt( np.sum(np.square(reference_nda - mean_ref)) / (N-1) )

            mean_stack = np.sum(stack_nda)/N
            std_stack = np.sqrt( np.sum(np.square(stack_nda - mean_stack)) / (N-1) )

            self._ncc[i] = np.sum((reference_nda - mean_ref) * (stack_nda - mean_stack)) / (N * std_ref * std_stack)
        
        return np.array(self._ncc)


    def _test_space(self, reference, stacks):
        ## Test that stacks and reference are in the same image space
        try:
            for i in range(0, self._N_stacks):
                stacks[i].sitk - reference.sitk
            
        except:
            raise ValueError("Reference and stack are not in the same space")