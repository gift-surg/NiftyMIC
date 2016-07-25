## \file StackManager.py
#  \brief  
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2015


## Import libraries
import os                       # used to execute terminal commands in python
import SimpleITK as sitk
import numpy as np

## Import modules from src-folder
import Stack as st


## StackManager contains all Stack objects and additional information which
#  shall enable an easier handling with stack-specific information
class StackManager:

    ## Constructor
    def __init__(self):
        self._stacks = None
        self._N_stacks = 0


    ## Constructor
    ## \param[in] stacks_to_copy list of Stack objects
    @classmethod
    def from_stacks(cls, stacks_to_copy):
        stack_manager = cls()

        N_stacks = len(stacks_to_copy)
        
        stacks = [None]*N_stacks
        for i in range(0, N_stacks):
            stack = st.Stack.from_stack(stacks_to_copy[i])
            stacks[i] = stack

        stack_manager._stacks = stacks
        stack_manager._N_stacks = N_stacks 

        return stack_manager


    ## Read input stacks and create list of Stack objects
    #  \param[in] dir_input directory where stacks are stored
    #  \param[in] filenames filenames of stacks to be considered in that directory
    #  \param[in] suffix_mask extension of stack filename which indicates associated mask
    def read_input_stacks(self, dir_input, filenames, suffix_mask):
        self._N_stacks = len(filenames)

        self._stacks = [None]*self._N_stacks

        for i in range(0, self._N_stacks):
            self._stacks[i] = st.Stack.from_filename(dir_input, filenames[i], suffix_mask)


    ## Read input stacks and create list of Stack objects
    #  \param[in] dir_input string to input directory where bundle of slices are stored
    #  \param[in] prefixes_stacks prefixes of stacks as list of strings indicating the corresponding stacks
    #  \param[in] suffix_mask extension of stack filename which indicates associated mask
    #  \example mask (suffix_mask) of slice j of stack i (prefix_stack) reads: i_j_mask.nii.gz
    def read_input_stacks_from_slices(self, dir_input, prefixes_stacks, suffix_mask):
        self._N_stacks = len(prefixes_stacks)

        self._stacks = [None]*self._N_stacks

        for i in range(0, self._N_stacks):
            self._stacks[i] = st.Stack.from_slice_filenames(dir_input, prefixes_stacks[i], suffix_mask)


    ## Get list of stored Stack objects
    #  \return List of Stack objects
    def get_stacks(self):
        return self._stacks

    
    ## Get number of Stack objects stored
    #  \return Number of Stacks, integer
    def get_number_of_stacks(self):
        return self._N_stacks


    ## Get affine transforms and corresponding rigid motion transform estimates 
    #  of each slice gathered during the entire evolution of the reconstruction
    #  algorithm.
    #  \return tuple of list of list of list of sitk.AffineTransform and sitk.Euler3DTransform objects
    #  \example return [affine_transforms, rigid_motion_transforms] with 
    #       affine_transforms[i][j][k] referring to the k-th estimated position,
    #       i.e. affine transform, of slice j of stack i and analog for
    #       rigid_motion_transforms
    def get_slice_registration_history_of_stacks(self):
        affine_transforms = [None]*self._N_stacks
        rigid_motion_transforms = [None]*self._N_stacks

        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()
            N_slices = stack.get_number_of_slices()

            affine_transforms[i] = [None]*N_slices
            rigid_motion_transforms[i] = [None]*N_slices
            
            for j in range(0, N_slices):
                slice = slices[j]
                slice_affine_transforms, slice_rigid_motion_transforms = slice.get_registration_history()

                N_cycles = len(slice_affine_transforms)

                affine_transforms[i][j] = [None]*N_cycles
                rigid_motion_transforms[i][j] = [None]*N_cycles

                for k in range(0, N_cycles):
                    affine_transforms[i][j][k] = slice_affine_transforms[k]
                    rigid_motion_transforms[i][j][k] = slice_rigid_motion_transforms[k]


        return affine_transforms, rigid_motion_transforms
        

    ## Get total amount of slices and begin of every new stack therein
    #  \return tuple containing total number of slices as integer and an array
    #       indicating the index of the first slice/beginning of each stack 
    def get_total_number_of_slices(self):
        N_slices_total = 0

        N_begin_new_stack_array = np.zeros(self._N_stacks)

        for i in range(0, self._N_stacks):
            N_slices_total += self._stacks[i].get_number_of_slices()
            N_begin_new_stack_array[i] = N_slices_total - self._stacks[i].get_number_of_slices()

        return N_slices_total, N_begin_new_stack_array


    ## Write all slices within all stacks (with current spatial transformations)
    #  to specified directory
    #  \param[in] directory directory where slices are written
    def write(self, directory):

        ## Write all slices
        for i in range(0, self._N_stacks):
            slices = self._stacks[i].get_slices()
            N_slices = self._stacks[i].get_number_of_slices()

            for j in range(0, N_slices):
                slices[j].write(directory=directory)

        print("All aligned slices successfully written to directory %s" % directory)


    ## Write all stacks to specified output directory
    #  to specified directory
    #  \param[in] directory directory where slices are written
    def write_stacks(self, directory):

        ## Write all slices
        for i in range(0, self._N_stacks):
            slices = self._stacks[i].write(directory=directory, filename=str(i), write_mask=True, write_slices=False)

