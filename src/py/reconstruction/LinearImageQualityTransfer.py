##-----------------------------------------------------------------------------
# \file LinearImageQualityTransfer.py
# \brief      Class deploying a linear model for image quality transfer
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Nov 2016


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import lsmr
from scipy.optimize import lsq_linear
from scipy.optimize import nnls
import time
from datetime import timedelta

## Import modules
import utilities.SimpleITKHelper as sitkh
import base.Stack as st


class LinearImageQualityTransfer(object):


    def __init__(self, stack, references, kernel_size=(6,6)):

        self._stack = st.Stack.from_stack(stack)
        self._N_slices = self._stack.get_number_of_slices()

        self._N_references = len(references)
        self._references = [None]*self._N_references

        for i in range(0, self._N_references):
            self._references[i] = st.Stack.from_stack(references[i])
    
        self._kernel_size = kernel_size


    def set_stack(self, stack):
        self._stack = st.Stack.from_stack(stack)


    def set_references(self, references):
        self._N_references = len(references)
        self._references = [None]*self._N_references

        for i in range(0, self._N_references):
            self._references[i] = st.Stack.from_stack(references[i])


    def set_kernel_size(self, kernel_size):
        self._kernel_size = kernel_size


    def get_kernel_size(self):
        return self._kernel_size


    def learn_linear_transfer(self):

        nda = sitk.GetArrayFromImage(self._stack.sitk)

        nda_references = [None]*self._N_references
        for i in range(0, self._N_references):
            nda_references = sitk.GetArrayFromImage(self._references[i].sitk)


        


