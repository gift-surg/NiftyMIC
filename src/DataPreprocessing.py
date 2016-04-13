## \file DataPreprocessing.py
#  \brief Performs preprocessing steps
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date April 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import numpy as np


## Import modules from src-folder
import SimpleITKHelper as sitkh
# import Stack as st
# import Slice as sl


## Class implementing data preprocessing steps
#  -# Crop stacks to region marked by mask
class DataPreprocessing:

    ## Constructor
    #  \param[in] dir_input directory where data is stored for preprocessing
    #  \param[in] dir_output directory in which preprocessed data gets written
    #  \param[in] target_stack_number in case only one mask is given (optional)
    def __init__(self, dir_input, dir_output, target_stack_number=0):

        self._dir_input = dir_input
        self._dir_output = dir_output
        self._target_stack_number = target_stack_number


    ## Perform data preprocessing step
    #  \param[in] filenames list of filenames referring to the data in dir_input to be processed
    #  \todo Mask propagation
    def run_preprocessing(self, filenames):
        N_stacks = len(filenames)

        number_of_masks = self._get_number_of_masks_in_directory()

        ## Each stack is provided a mask
        if number_of_masks is N_stacks:
            print("All given stacks and masks are cropped to their masked region.")

            for i in range(0, N_stacks):
                ## Read stack and mask from directory
                stack_sitk = sitk.ReadImage(self._dir_input + filenames[i] + ".nii.gz")
                mask_sitk = sitk.ReadImage(self._dir_input + filenames[i] + "_mask.nii.gz", sitk.sitkUInt8)

                ## Crop stack and mask based on the mask provided
                [stack_sitk, mask_sitk] = self._crop_stack_and_mask(stack_sitk, mask_sitk)

                ## Write preprocessed data to output directory
                self._write_preprocessed_stack_and_mask(stack_sitk, mask_sitk, filenames[i])

        ## No stack is provided a mask. Hence, mask entire region of stack
        elif number_of_masks is 0:
            print("No mask is provided. Consider entire stack for reconstruction pipeline.")

            for i in range(0, N_stacks):
                ## Read stack from directory
                stack_sitk = sitk.ReadImage(self._dir_input + filenames[i] + ".nii.gz")

                ## Create binary mask consisting of ones
                shape = sitk.GetArrayFromImage(stack_sitk).shape
                nda = np.ones(shape, dtype=np.uint8)

                mask_sitk = sitk.GetImageFromArray(nda)
                mask_sitk.CopyInformation(stack_sitk) 
            
                ## Write preprocessed data to output directory
                self._write_preprocessed_stack_and_mask(stack_sitk, mask_sitk, filenames[i])

        ## Not all stacks are provided a mask. Propagate target stack mask to other stacks
        ## \todo Mask propagation
        else:
            print("Not all stacks are provided a mask. Mask of target stack is propagated to other masks.")
            # if os.path.isfile(self._dir_input + filenames[i] + "_mask.nii.gz"):
            raise ValueError("Error: Mask propagation not provided yet.")


    ## Count number of masks given in directory
    #  \return number of masks found in directory
    def _get_number_of_masks_in_directory(self):

        number_of_masks = 0

        ## List of all files in directory
        all_files = os.listdir(self._dir_input)

        ## Count number of files labelled as masks
        for file in all_files:

            if file.endswith("_mask.nii.gz"):
                number_of_masks += 1

        return number_of_masks


    ## Crop stack and mask to region given my mask
    #  \param[in] stack_sitk stack as sitk.Image object
    #  \param[in] mask_sitk mask as sitk.Image object
    #  \return cropped stack as sitk.Object
    #  \return cropped mask as sitk.Object
    def _crop_stack_and_mask(self, stack_sitk, mask_sitk):

        ## Get rectangular region surrounding the masked voxels
        [x_range, y_range, z_range] = self._get_rectangular_masked_region(mask_sitk, boundary=0)

        ## Crop stack and mask to defined image region
        stack_crop_sitk = self._crop_image_to_region(stack_sitk, x_range, y_range, z_range)
        mask_crop_sitk = self._crop_image_to_region(mask_sitk, x_range, y_range, z_range)

        return stack_crop_sitk, mask_crop_sitk
        

    ## Return rectangular region surrounding masked region. 
    #  \param[in] mask_sitk sitk.Image representing the mask
    #  \param[in] boundary additional boundary surrounding mask (optional)
    #  \return range_x pair defining x interval of mask in voxel space 
    #  \return range_y pair defining y interval of mask in voxel space
    #  \return range_z pair defining z interval of mask in voxel space
    def _get_rectangular_masked_region(self, mask_sitk, boundary=0):

        ## Get mask array
        nda = sitk.GetArrayFromImage(mask_sitk)
        
        ## Get shape defining the dimension in each direction
        shape = nda.shape

        ## Set additional offset around identified masked region
        offset_x = boundary
        offset_y = boundary
        offset_z = boundary

        ## Compute sum of pixels of each slice along specified directions
        sum_xy = np.sum(nda, axis=(0,1)) # sum within x-y-plane
        sum_xz = np.sum(nda, axis=(0,2)) # sum within x-z-plane
        sum_yz = np.sum(nda, axis=(1,2)) # sum within y-z-plane

        ## Find masked regions (non-zero sum!)
        range_x = np.zeros(2)
        range_y = np.zeros(2)
        range_z = np.zeros(2)

        ## Non-zero elements of numpy array nda defining x_range
        ran = np.nonzero(sum_yz)[0]
        range_x[0] = np.max( [0,         ran[0]-offset_x] )
        range_x[1] = np.min( [shape[0], ran[-1]+offset_x+1] )

        ## Non-zero elements of numpy array nda defining y_range
        ran = np.nonzero(sum_xz)[0]
        range_y[0] = np.max( [0,         ran[0]-offset_y] )
        range_y[1] = np.min( [shape[1], ran[-1]+offset_y+1] )

        ## Non-zero elements of numpy array nda defining z_range
        ran = np.nonzero(sum_xy)[0]
        range_z[0] = np.max( [0,         ran[0]-offset_z] )
        range_z[1] = np.min( [shape[2], ran[-1]+offset_z+1] )

        ## Numpy reads the array as z,y,x coordinates! So swap them accordingly
        return range_z.astype(int), range_y.astype(int), range_x.astype(int)


    ## Crop given image to region defined by voxel space ranges
    #  \param[in] image_sitk image which will be cropped
    #  \param[in] range_x pair defining x interval in voxel space for image cropping
    #  \param[in] range_y pair defining y interval in voxel space for image cropping
    #  \param[in] range_z pair defining z interval in voxel space for image cropping
    #  \return image cropped to defined region
    def _crop_image_to_region(self, image_sitk, range_x, range_y, range_z):

        image_cropped_sitk = image_sitk[\
                                range_x[0]:range_x[1],\
                                range_y[0]:range_y[1],\
                                range_z[0]:range_z[1]\
                            ]

        return image_cropped_sitk


    ## Write preprocessed stack and mask to given output folder
    #  \param[in] stack_sitk stack to be written
    #  \param[in] stack_sitk mask to be written
    #  \param[in] filename filename to be used
    def _write_preprocessed_stack_and_mask(self, stack_sitk, mask_sitk, filename):

        ## Write stack
        sitk.WriteImage(stack_sitk, self._dir_output + filename + ".nii.gz")

        ## Write mask
        sitk.WriteImage(mask_sitk, self._dir_output + filename + "_mask.nii.gz")

