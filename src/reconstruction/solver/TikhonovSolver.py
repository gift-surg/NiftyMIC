#!/usr/bin/python

## \file TikhonovSolver.py
#  \brief Implementation to get an approximate solution of the inverse problem 
#  \f$ y_k = A_k x \f$ for each slice \f$ y_k,\,k=1,\dots,K \f$
#  by using Tikhonov-regularization
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date July 2016

## Import libraries
import os                       # used to execute terminal commands in python
import sys
import itk
import SimpleITK as sitk
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lsqr
from scipy.optimize import lsq_linear

## Add directories to import modules
dir_src_root = "../src/"
sys.path.append( dir_src_root + "base/" )
sys.path.append( dir_src_root + "reconstruction/" )
sys.path.append( dir_src_root + "reconstruction/solver/" )

## Import modules
import SimpleITKHelper as sitkh
import DifferentialOperations as diffop
from Solver import Solver

## Pixel type of used 3D ITK image
PIXEL_TYPE = itk.D

## ITK image type 
IMAGE_TYPE = itk.Image[PIXEL_TYPE, 3]


class TikhonovSolver(Solver):

    ## Constructor
    #  \param[in] stacks list of Stack objects containing all stacks used for the reconstruction
    #  \param[in] HR_volume Stack object containing the current estimate of the HR volume (used as initial value + space definition)
    #  \param[in] alpha_cut Cut-off distance for Gaussian blurring filter
    def __init__(self, stacks, HR_volume, alpha_cut=3, alpha=0.03, iter_max=20):

        Solver.__init__(self, stacks, HR_volume, alpha_cut)

        ## Compute total amount of pixels for all slices
        N_total_slice_voxels = 0
        for i in range(0, self._N_stacks):
            N_stack_voxels = np.array(self._stacks[i].sitk.GetSize()).prod()
            N_total_slice_voxels += N_stack_voxels

        ## Compute total amount of voxels to store for Ax:
        self._N_voxels_HR_volume = np.array(self._HR_volume.sitk.GetSize()).prod()
        self._N_total_voxels = N_total_slice_voxels + 3*self._N_voxels_HR_volume

        ## Define differential operators
        spacing = self._HR_volume.sitk.GetSpacing()[0]
        self._differential_operations = diffop.DifferentialOperations(step_size=spacing, Laplace_comp_type="FiniteDifference")                  
        
        ## Settings for optimizer
        self._alpha = alpha
        self._iter_max = iter_max


    def run_reconstruction(self):

        A_fw = lambda x: self._A_fw(x, self._alpha)
        A_bw = lambda x: self._A_bw(x, self._alpha)
        HR_nda = sitk.GetArrayFromImage(self._HR_volume.sitk)

        A = LinearOperator((self._N_total_voxels, self._N_voxels_HR_volume), matvec=A_fw, rmatvec=A_bw)
        b = self._b()

        # res = lsq_linear(A, b, bounds=(0, np.inf), max_iter=10, lsq_solver=None, lsmr_tol='auto', verbose=2)
        res = lsqr(A, b, iter_lim=self._iter_max, show=True)

        HR_volume_itk = self._get_HR_itk_image_from_array_vec(res[0])        
        sitkh.show_itk_image(HR_volume_itk)        


        return res
        # A.rmatvec()



    def _b(self):
        b = np.zeros(self._N_total_voxels)

        i_min = 0
        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()

            N_slice_voxels = np.array(slices[0].sitk.GetSize()).prod()

            for k in range(0, stack.get_number_of_slices()):
                i_max = i_min + N_slice_voxels

                slice_k = slices[k]

                slice_itk = self.Mk(slice_k.itk, slice_k)
                slice_nda = self._itk2np.GetArrayFromImage(slice_itk)

                b[i_min:i_max] = slice_nda.flatten()

                i_min = i_max

        return b



    def _A_fw(self, HR_nda_vec, alpha):
        
        x_itk = self._get_HR_itk_image_from_array_vec(HR_nda_vec)

        A_x = np.zeros(self._N_total_voxels)

        i_min = 0
        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()

            N_slice_voxels = np.array(slices[0].sitk.GetSize()).prod()

            for k in range(0, stack.get_number_of_slices()):
                i_max = i_min + N_slice_voxels

                slice_k = slices[k]

                slice_itk = self.Mk_Ak(x_itk, slice_k)
                slice_nda = self._itk2np.GetArrayFromImage(slice_itk)

                A_x[i_min:i_max] = slice_nda.flatten()

                i_min = i_max

        A_x[i_min:] = self._D(HR_nda_vec)*np.sqrt(alpha)

        return A_x



    def _D(self, HR_nda_vec):
        
        N_voxels = self._N_voxels_HR_volume
        D_x = np.zeros(3*N_voxels)

        HR_nda = HR_nda_vec.reshape( self._HR_shape_nda )

        D_x[0:N_voxels] = self._differential_operations.Dx(HR_nda).flatten()
        D_x[N_voxels:2*N_voxels] = self._differential_operations.Dy(HR_nda).flatten()
        D_x[2*N_voxels:3*N_voxels] = self._differential_operations.Dz(HR_nda).flatten()

        return D_x


    def _A_bw(self, slice_nda_vec, alpha):

        A_adj_y = np.zeros(self._N_voxels_HR_volume)

        i_min = 0

        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()

            N_slice_voxels = np.array(slices[0].sitk.GetSize()).prod()

            for k in range(0, stack.get_number_of_slices()):
                # print("(i,k) = (%s,%s)" %(i,k))
                i_max = i_min + N_slice_voxels

                slice_k = slices[k]

                slice_itk = self._get_itk_image_from_array_vec( slice_nda_vec[i_min:i_max], slice_k.itk )

                tmp_itk = self.Ak_adj_Mk(slice_itk, slice_k)
                tmp_nda = self._itk2np.GetArrayFromImage(tmp_itk).flatten()

                A_adj_y = A_adj_y + tmp_nda

                i_min = i_max

        A_adj_y = A_adj_y + self._D_adj(slice_nda_vec).flatten()*np.sqrt(alpha)

        return A_adj_y



    def _D_adj(self, slice_nda_vec):

        N_voxels = self._N_voxels_HR_volume
        D_adj_slice = np.zeros(3*N_voxels)

        slice_x_nda_vec = slice_nda_vec[-3*N_voxels: -2*N_voxels]
        slice_y_nda_vec = slice_nda_vec[-2*N_voxels: -N_voxels]
        slice_z_nda_vec = slice_nda_vec[-N_voxels:]

        slice_x_nda = slice_x_nda_vec.reshape( self._HR_shape_nda )
        slice_y_nda = slice_y_nda_vec.reshape( self._HR_shape_nda )
        slice_z_nda = slice_z_nda_vec.reshape( self._HR_shape_nda )

        return self._differential_operations.Dx_adj(slice_x_nda) + self._differential_operations.Dy_adj(slice_y_nda) + self._differential_operations.Dz_adj(slice_z_nda)



                
            