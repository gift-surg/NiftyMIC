## \file VolumeReconstruction.py
#  \brief Reconstruct volume given the current position of slices 
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date December 2015


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import numpy as np


## Import modules from src-folder
import SimpleITKHelper as sitkh
# import Stack as st
# import Slice as sl


## Class implementing the volume reconstruction given the current position of slices
class VolumeReconstruction:

    ## Constructor
    #  \param stack_manager instance of StackManager containing all stacks and additional information
    def __init__(self, stack_manager):
        self._stack_manager = stack_manager
        self._stacks = stack_manager.get_stacks()
        self._N_stacks = stack_manager.get_number_of_stacks()



    ## Computed reconstructed volume based on current estimated positions of slices
    #  \param[in,out] HR_volume current estimate of reconstructed HR volume (Stack object)
    def update_reconstructed_volume(self, HR_volume):
        print("Update reconstructed volume")

        self._use_discrete_shepard(HR_volume)


    ## Recontruct volume based on discrete Shepard's like method, cf. Vercauteren2006, equation (19)
    #  \param[in,out] HR_volume current estimate of reconstructed HR volume (Stack object)
    def _use_discrete_shepard(self, HR_volume):
        sigma = 1

        shape = sitk.GetArrayFromImage(HR_volume.sitk).shape
        helper_N_nda = np.zeros(shape)
        helper_D_nda = np.zeros(shape)

        default_pixel_value = 0.0

        for i in range(0, self._N_stacks):
        # for i in range(0, 2):
            print("  Stack %s/%s" %(i,self._N_stacks-1))
            stack = self._stacks[i]
            slices = stack.get_slices()
            N_slices = stack.get_number_of_slices()
            
            for j in range(0, N_slices):
                slice = slices[j]

                ## Nearest neighbour resampling of slice to target space (HR volume)
                slice_resampled_sitk = sitk.Resample(
                    slice.sitk, 
                    HR_volume.sitk, 
                    sitk.Euler3DTransform(), 
                    sitk.sitkNearestNeighbor, 
                    default_pixel_value,
                    HR_volume.sitk.GetPixelIDValue())

                ## Extract array of pixel intensities
                nda_slice = sitk.GetArrayFromImage(slice_resampled_sitk)

                ## Look for indices which are stroke by the slice in the isotropic grid
                ind_nonzero = nda_slice>0

                ## update arrays of numerator and denominator
                helper_N_nda[ind_nonzero] += nda_slice[ind_nonzero]
                helper_D_nda[ind_nonzero] += 1
                
                # print("helper_N_nda: (min, max) = (%s, %s)" %(np.min(helper_N_nda), np.max(helper_N_nda)))
                # print("helper_D_nda: (min, max) = (%s, %s)" %(np.min(helper_D_nda), np.max(helper_D_nda)))


        ## Create sitk-images with correct header data
        helper_N = sitk.GetImageFromArray(helper_N_nda) 
        helper_D = sitk.GetImageFromArray(helper_D_nda) 

        helper_N.CopyInformation(HR_volume.sitk)
        helper_D.CopyInformation(HR_volume.sitk)

        ## Apply recursive Gaussian smoothing
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(sigma)

        HR_volume_update_N = gaussian.Execute(helper_N)
        HR_volume_update_D = gaussian.Execute(helper_D)

        ## Avoid undefined division by zero
        """
        HACK start
        """
        ## HACK for denominator
        nda = sitk.GetArrayFromImage(HR_volume_update_D)
        ind_min = np.unravel_index(np.argmin(nda), nda.shape)
        # print nda[nda<0]
        # print nda[ind_min]

        eps = 1e-8
        nda[nda<=eps]=1   # actually only nda=0 but some are negative!?
        print("denominator min = %s" % np.min(nda))


        HR_volume_update_D = sitk.GetImageFromArray(nda)
        HR_volume_update_D.CopyInformation(HR_volume.sitk)

        ## HACK for numerator given that some intensities are negative!?
        nda = sitk.GetArrayFromImage(HR_volume_update_N)
        ind_min = np.unravel_index(np.argmin(nda), nda.shape)
        nda[nda<=eps]=0
        # print nda[nda<0]
        print("numerator min = %s" % np.min(nda))
        """
        HACK end
        """
        
        ## Compute HR volume based on scattered data approximation with correct header (might be redundant):
        HR_volume_update = HR_volume_update_N/HR_volume_update_D
        HR_volume_update.CopyInformation(HR_volume.sitk)

        ## Update HR volume image file within Stack-object HR_volume
        HR_volume.sitk = HR_volume_update


        ##
        nda = sitk.GetArrayFromImage(HR_volume_update)
        print np.min(nda)

        HR_volume.show()
