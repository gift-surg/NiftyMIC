#!/usr/bin/python

## \file differential_operations.py
#  \brief Implementation of 3D differential operations
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date July 2016

## Import libraries
import os
import sys
import numpy as np
from scipy import ndimage

## Class to implement differential operations used in solver class
class DifferentialOperations:

    ##
    # Create differential kernel operators and store as member attributes.
    # Note: kernel = np.zeros((z,y,x)) in numpy!
    # \param[in]  step_size          step size for all discrete
    #                                differentiations
    # \param[in]  Laplace_comp_type  can be either 'LaplaceStencil' or
    #                                'FiniteDifference'
    # \param[in]  convolution_mode   Periodic/Circular boundary conditions
    #                                ('wrap'). Zero boundary conditions
    #                                ('constant'). See scipy help page.
    #
    def __init__(self, step_size=1, Laplace_comp_type="FiniteDifference", convolution_mode="constant"):

        ## step size for all discrete differentiations
        self._step_size = step_size

        ## Convolution mode
        self._convolution_mode = convolution_mode

        ## Computational type of Laplacian compuation
        if Laplace_comp_type not in ["LaplaceStencil", "FiniteDifference"]:
            raise ValueError("Error: Laplace computation type can only be either 'LaplaceStencil' or 'FiniteDifference'")

        else:
            self._Laplace_comp_type = Laplace_comp_type
        
        self._Laplace = {
            "LaplaceStencil"    :   self._Laplace_stencil,
            "FiniteDifference"  :   self._Laplace_FD
        }

        """
        Difference Quotients to differentiate array
        with array[z,y,x], i.e. the 'correct' direction by viewing the 
        resulting nifti-image differentiation. The resulting kernel
        can be used via _convolve(kernel, nda) to differentiate image
        """
        ## Forward differential quotient in x-direction
        self._KERNEL_DX_FW = np.zeros((1,1,2))
        self._KERNEL_DX_FW[:] = np.array([1,-1])

        ## Backward differential quotient in x-direction
        self._KERNEL_DX_BW = np.zeros((1,1,3))
        self._KERNEL_DX_BW[:] = np.array([0,1,-1])

        ## Forward differential quotient in y-direction
        self._KERNEL_DY_FW = np.zeros((1,2,1))
        self._KERNEL_DY_FW[:] = np.array([[1],[-1]])

        ## Backward differential quotient in y-direction
        self._KERNEL_DY_BW = np.zeros((1,3,1))
        self._KERNEL_DY_BW[:] = np.array([[0],[1],[-1]])

        ## Forward differential quotient in z-direction
        self._KERNEL_DZ_FW = np.zeros((2,1,1))
        self._KERNEL_DZ_FW[:] = np.array([[[1]],[[-1]]])

        ## Backward differential quotient in z-direction
        self._KERNEL_DZ_BW = np.zeros((3,1,1))
        self._KERNEL_DZ_BW[:] = np.array([[[0]],[[1]],[[-1]]])

        ## 3D Laplacian self._kernel based on 5-point stencil in 2D
        self._KERNEL_L_STENCIL = np.zeros((3,3,3))
        self._KERNEL_L_STENCIL[0,1,1] = 1
        self._KERNEL_L_STENCIL[1,:,:] = np.array([[0, 1, 0],[1, -6, 1],[0, 1, 0]])
        self._KERNEL_L_STENCIL[2,1,1] = 1


    ## Set step size for difference quotient
    #  \param[in] step_size step size for all discrete differentiations
    def set_step_size(self, step_size):
        self._step_size = step_size


    ## Forward differentiation in x
    #  \param[in] nda data array of 3D image as numpy array
    #  \return differentiated numpy array
    def Dx(self, nda):
        kernel = self._KERNEL_DX_FW / self._step_size
        return ndimage.convolve(nda, kernel, mode=self._convolution_mode)


    ## Forward differentiation in x
    #  Periodic/Circular boundary conditions are used ('wrap')
    #  \param[in] nda data array of 3D image as numpy array
    #  \return differentiated numpy array
    def Dx_adj(self, nda):
        kernel = -self._KERNEL_DX_BW / self._step_size
        return ndimage.convolve(nda, kernel, mode=self._convolution_mode)


    ## Forward differentiation in y
    #  Periodic/Circular boundary conditions are used ('wrap')
    #  \param[in] nda data array of 3D image as numpy array
    #  \return differentiated numpy array
    def Dy(self, nda):
        kernel = self._KERNEL_DY_FW / self._step_size
        return ndimage.convolve(nda, kernel, mode=self._convolution_mode)


    ## Forward differentiation in y
    #  Periodic/Circular boundary conditions are used ('wrap')
    #  \param[in] nda data array of 3D image as numpy array
    #  \return differentiated numpy array
    def Dy_adj(self, nda):
        kernel = -self._KERNEL_DY_BW / self._step_size
        return ndimage.convolve(nda, kernel, mode=self._convolution_mode)


    ## Forward differentiation in z
    #  Periodic/Circular boundary conditions are used ('wrap')
    #  \param[in] nda data array of 3D image as numpy array
    #  \return differentiated numpy array
    def Dz(self, nda):
        kernel = self._KERNEL_DZ_FW / self._step_size
        return ndimage.convolve(nda, kernel, mode=self._convolution_mode)


    ## Forward differentiation in z
    #  Periodic/Circular boundary conditions are used ('wrap')
    #  \param[in] nda data array of 3D image as numpy array
    #  \return differentiated numpy array
    def Dz_adj(self, nda):
        kernel = -self._KERNEL_DZ_BW / self._step_size
        return ndimage.convolve(nda, kernel, mode=self._convolution_mode)

    ## Laplacian operation
    #  Periodic/Circular boundary conditions are used ('wrap')
    #  \param[in] nda data array of 3D image as numpy array
    #  \return differentiated numpy array
    def Laplace(self, nda):
        return self._Laplace[self._Laplace_comp_type](nda)


    ## Laplacian operation based on Laplace stencil
    #  Periodic/Circular boundary conditions are used ('wrap')
    #  \param[in] nda data array of 3D image as numpy array
    #  \return differentiated numpy array
    def _Laplace_stencil(self, nda):
        kernel = self._KERNEL_L_STENCIL / self._step_size * self._step_size
        return ndimage.convolve(nda, kernel, mode=self._convolution_mode)


    ## Laplacian operation based on composite finite differences, i.e.
    #  \f[
    #   \Delta f = \nabla^T \nabla f = [D_x; D_y; D_z]^T [D_x; D_y; D_z] f
    #           = [D_x^T D_x + D_y^T D_y + D_z^T D_z] f
    #  \f]
    #  Periodic/Circular boundary conditions are used ('wrap')
    #  \param[in] nda data array of 3D image as numpy array
    #  \return differentiated numpy array
    def _Laplace_FD(self, nda):

        ## Forward Operation
        Dx = self.Dx(nda)
        Dy = self.Dy(nda)
        Dz = self.Dz(nda)

        ## Adjoint Operation
        DTDx = self.Dx_adj(Dx)
        DTDy = self.Dy_adj(Dy)
        DTDz = self.Dz_adj(Dz)

        return DTDx + DTDy + DTDz
