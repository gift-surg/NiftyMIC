## \file StackManager.py
#  \brief  
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2015


## Import libraries
import os                       # used to execute terminal commands in python
import SimpleITK as sitk

## Import modules from src-folder
import Stack as st

## StackManager contains all Stack objects and additional information which
#  shall enable an easier handling with stack-specific information
class StackManager:

    def __init__(self):
        self._stacks = None
        self._N_stacks = 0


    ## Read input stacks and create list of Stack objects
    #  \param[in] dir_input directory where stacks are stored
    #  \param[in] filenames filenames of stacks to be considered in that directory
    def read_input_stacks(self, dir_input, filenames):
        self._N_stacks = len(filenames)

        self._stacks = [None]*self._N_stacks

        for i in range(0, self._N_stacks):
            self._stacks[i] = st.Stack.from_nifti(dir_input, filenames[i])


    ## Get list of stored Stack objects
    #  \return List of Stack objects
    def get_stacks(self):
        return self._stacks

    
    ## Get number of Stack objects stored
    #  \return Number of Stacks, integer
    def get_number_of_stacks(self):
        return self._N_stacks


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

