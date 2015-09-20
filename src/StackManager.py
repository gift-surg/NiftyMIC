## \file StackManager.py
#  \brief  
# 
#  \author Michael Ebner
#  \date September 2015


## Import libraries
import os                       # used to execute terminal commands in python
import SimpleITK as sitk

## Import modules from src-folder
import Stack as stack

class StackManager:

    def __init__(self):
        return None


    def read_input_data(self, dir_input, filenames):
        self._N_stacks = len(filenames)

        self._stacks = [None]*self._N_stacks

        for i in range(0, self._N_stacks):
            self._stacks[i] = stack.Stack(dir_input, filenames[i])


    def get_stacks(self):
        return self._stacks


    def get_number_of_stacks(self):
        return self._N_stacks


    def write_results(self, directory):
        # for i in range(0, self._N_stacks):
        for i in range(0, 1):
            slices = self._stacks[i].get_slices()
            N_slices = self._stacks[i].get_number_of_slices()

            for j in range(0, N_slices):
                slices[j].write_results(directory)