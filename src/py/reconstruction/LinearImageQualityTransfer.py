##
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
from scipy.optimize import least_squares
from scipy.optimize import nnls
import time
from datetime import timedelta

## Import modules
import utilities.SimpleITKHelper as sitkh
import base.Stack as st


class LinearImageQualityTransfer(object):


    def __init__(self, stack, references=None, kernel_shape=(6,6), convolution_mode="constant"):

        self._stack = st.Stack.from_stack(stack)
        self._N_slices = self._stack.get_number_of_slices()

        if references is not None:
            self._N_references = len(references)
            self._references = [None]*self._N_references

            for i in range(0, self._N_references):
                self._references[i] = st.Stack.from_stack(references[i])
    
        self._kernel_shape = np.array(kernel_shape)
        self._convolution_mode = convolution_mode


    def set_stack(self, stack):
        self._stack = st.Stack.from_stack(stack)


    def set_references(self, references):
        self._N_references = len(references)
        self._references = [None]*self._N_references

        for i in range(0, self._N_references):
            self._references[i] = st.Stack.from_stack(references[i])


    def set_kernel_shape(self, kernel_shape):
        self._kernel_shape = np.array(kernel_shape)


    def get_kernel_shape(self):
        return self._kernel_shape


    def get_reconstructed_stack(self):
        return st.Stack.from_stack(self._stack)


    def get_kernel(self):
        return np.array(self._kernel)


    def set_kernel(self, kernel):
        self._kernel = np.array(kernel)

    ##
    # \date       2016-07-29 12:30:30+0100
    # \brief      Print statistics associated to performed reconstruction
    #
    # \param[in]  self  The object
    #
    def print_statistics(self):
        # print("\nStatistics for performed registration:" %(self._reg_type))
        # if self._elapsed_time_sec < 0:
        #     raise ValueError("Error: Elapsed time has not been measured. Run 'run_reconstruction' first.")
        # else:
        print("\tElapsed time: %s" %(timedelta(seconds=self._elapsed_time_sec)))
        # print("\tell^2-residual sum_k ||M_k(A_k x - y_k||_2^2 = %.3e" %(self._residual_ell2))
        # print("\tprior residual = %.3e" %(self._residual_prior))


    ##
    #       Apply linear transfer model, i.e. A(kernel)x
    # \date       2016-11-06 05:10:30+0000
    #
    # \param      self   The object
    # \param      stack  The stack on which the convolution shall be applied as
    #                    Stack object
    #
    def apply_linear_quality_transfer(self, stack=None):

        if stack is None:
            stack = self._stack

        nda = sitk.GetArrayFromImage(stack.sitk)

        for i in range(0, nda.shape[0]):
            nda[i,:,:] = ndimage.convolve(nda[i,:,:], self._kernel, mode=self._convolution_mode)

        stack_sitk = sitk.GetImageFromArray(nda)
        stack_sitk.CopyInformation(stack.sitk)

        self._stack = st.Stack.from_sitk_image(stack_sitk, stack.get_filename()+"_liqt")


    ##
    #       Learn linear mapping, i.e. kernel coefficients, so that y_PD
    #             = A(kernel)x
    # \date       2016-11-06 05:11:18+0000
    #
    # \param      self  The object
    #
    def learn_linear_transfer(self):

        ## Get mask of first reference for focus
        nda_mask = sitk.GetArrayFromImage(self._references[0].sitk_mask)        

        ## Get data array x
        self._nda = sitk.GetArrayFromImage(self._stack.sitk)*nda_mask

        ## Get arrays of references to 'learn' transfer
        self._nda_references = [None]*self._N_references
        for i in range(0, self._N_references):
            self._nda_references[i] = sitk.GetArrayFromImage(self._references[i].sitk) * nda_mask

        ## Get dimensions for residual and jacobian
        self._N_residual = self._nda.size * self._N_references
        self._N_parameters = self._kernel_shape.prod()

        ## Compute (constant) jacobian:
        jacobian = self._get_jacobian_residual_data_fit(np.ones(self._kernel_shape.prod()))
        
        ## Define variables for least-squares optimization        
        kernel0 = np.ones(self._kernel_shape).flatten()
        fun = lambda x: self._get_residual_data_fit(x)
        jac = lambda x: jacobian

        ## Time optimization
        time_start = time.time()

        ## Non-linear least-squares method:
        # res = least_squares(fun=fun, x0=kernel0, method='trf', verbose=2) 
        res = least_squares(fun=fun, x0=kernel0, jac=jac, method='trf', loss='linear',verbose=2) 
        # res = least_squares(fun=fun, x0=kernel0, jac='2-point', method='trf', verbose=verbose) 
        # res = least_squares(fun=fun, x0=kernel0, method='lm', loss='linear', tr_solver='exact', verbose=1) 
        self._kernel = res.x.reshape(self._kernel_shape)

        ## Set elapsed time
        time_end = time.time()
        self._elapsed_time_sec = time_end-time_start


    ##
    #       Compute residual y_PD - A(theta)x based on parameters
    #             theta representing the kernel coefficients
    # \date       2016-11-06 05:03:31+0000
    #
    # \param      self        The object
    # \param      kernel_vec  The kernel vector
    #
    # \return     The residual data fit as N-array
    #
    def _get_residual_data_fit(self, kernel_vec):

        ## Allocate memory
        residual = np.zeros((self._N_references, self._nda.size))

        ## Expand to 3D for easier computation (faster too?)
        kernel_3D = kernel_vec.reshape(1, self._kernel_shape[0], self._kernel_shape[1])

        ## Compute A(theta)x
        nda_convolved = ndimage.convolve(self._nda, kernel_3D, mode=self._convolution_mode)

        ## Compute y_PD_i - A(theta)x
        for i in range(0, self._N_references):
            residual[i,:] = (self._nda_references[i]-nda_convolved).flatten()
    
        return residual.flatten()


    ##
    #       Gets the jacobian residual data fit. Given the structure, the
    #             jacobian is constant.
    # \date       2016-11-06 05:06:15+0000
    #
    # \param      self        The object
    # \param      kernel_vec  The kernel vector
    #
    # \return     The jacobian residual data fit.
    #
    def _get_jacobian_residual_data_fit(self, kernel_vec):
        
        ## Allocate memory
        jacobian = np.zeros((self._N_residual, self._N_parameters))
        
        ## Get derivative w.r.t to parameters
        for j in range(0, self._N_parameters):
            
            ## Kernel multiplication is linear function, hence only a single
            ## coefficient remains
            kernel = np.zeros_like(kernel_vec)
            kernel[j] = 1
            kernel_3D = kernel.reshape(1, self._kernel_shape[0], self._kernel_shape[1])

            ## Compute derivative
            nda_convolved = ndimage.convolve(self._nda, kernel_3D, mode=self._convolution_mode)
            jacobian[:,j] = -np.tile(nda_convolved.flatten(), self._N_references)

        return jacobian



