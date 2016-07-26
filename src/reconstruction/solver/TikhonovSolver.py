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
from scipy.optimize import nnls

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


## This class implements the framework to iteratively solve 
#  \f$ \vec{y}_k = A_k \vec{x} \f$ for every slice \f$ \vec{y}_k,\,k=1,\dots,K \f$
#  via Tikhonov regularization via an augmented least-square approach
#  where \f$A_k=D_k B_k\f$ denotes the combined blurring and downsampling 
#  operation, \f$ M_k \f$ the masking operator and \f$G\f$ represents either 
#  the identity matrix \f$I\f$ (zeroth-order Tikhonov) or 
#  the (flattened, stacked vector) gradient 
#  \f$ \nabla  = \begin{pmatrix} D_x \\ D_y \\ D_z \end{pmatrix} \f$ 
#  (first-order Tikhonov).
#  The minimization problem reads
#  \f[
#       \text{arg min}_{\vec{x}} \Big( \sum_{k=1}^K \frac{1}{2} \Vert M_k (\vec{y}_k - A_k \vec{x} )\Vert_{\ell^2}^2 
#                       + \frac{\alpha}{2}\,\Vert G\vec{x} \Vert_{\ell^2}^2 \Big)
#       = 
#       \text{arg min}_{\vec{x}} \Bigg( \Bigg\Vert 
#           \begin{pmatrix} M_1 A_1 \\ M_2 A_2 \\ \vdots \\ M_K A_K \\ \sqrt{\alpha} G \end{pmatrix} \vec{x}
#           - \begin{pmatrix}  M_1 \vec{y}_1 \\ M_2 \vec{y}_2 \\ \vdots \\ M_K \vec{y}_K \\ \vec{0} \end{pmatrix}
#       \Bigg\Vert_{\ell^2}^2 \Bigg)
#  \f] 
#  \see \p itkAdjointOrientedGaussianInterpolateImageFilter of \p ITK
#  \see \p itOrientedGaussianInterpolateImageFunction of \p ITK
class TikhonovSolver(Solver):

    ## Constructor
    #  \param[in] stacks list of Stack objects containing all stacks used for the reconstruction
    #  \param[in] HR_volume Stack object containing the current estimate of the HR volume (used as initial value + space definition)
    #  \param[in] alpha_cut Cut-off distance for Gaussian blurring filter
    def __init__(self, stacks, HR_volume, alpha_cut=3, alpha=0.02, iter_max=10, reg_type="TK1"):

        Solver.__init__(self, stacks, HR_volume, alpha_cut)

        ## Compute total amount of pixels for all slices
        self._N_total_slice_voxels = 0
        for i in range(0, self._N_stacks):
            N_stack_voxels = np.array(self._stacks[i].sitk.GetSize()).prod()
            self._N_total_slice_voxels += N_stack_voxels

        ## Compute total amount of voxels of x:
        self._N_voxels_HR_volume = np.array(self._HR_volume.sitk.GetSize()).prod()
        # self._N_total_voxels = N_total_slice_voxels + 3*self._N_voxels_HR_volume

        ## Define differential operators
        spacing = self._HR_volume.sitk.GetSpacing()[0]
        self._differential_operations = diffop.DifferentialOperations(step_size=spacing)                  
        
        ## Settings for optimizer
        self._alpha = alpha
        self._iter_max = iter_max
        self._reg_type = reg_type

        self._A = {
            "TK0"   : self._A_TK0,
            "TK1"   : self._A_TK1
        }

        self._A_adj = {
            "TK0"   : self._A_adj_TK0,
            "TK1"   : self._A_adj_TK1
        }

    ## Set type of regularization. It can be either 'TK0' or 'TK1'
    #  \param[in] reg_type Either 'TK0' or 'TK1', string
    def set_regularization_type(self, reg_type):
        if reg_type not in ["TK0", "TK1"]:
            raise ValueError("Error: regularization type can only be either 'TK0' or 'TK1'")

        self._reg_type = reg_type


    ## Get chosen type of regularization.
    #  \return regularization type as string
    def get_regularization_type(self):
        return self._reg_type


    ## Set regularization parameter
    #  \param[in] alpha regularization parameter, scalar
    def set_alpha(self, alpha):
        self._alpha = alpha


    ## Get value of chosen regularization parameter
    #  \return regularization parameter, scalar
    def get_alpha(self):
        return self._alpha


    ## Set maximum number of iterations for minimizer
    #  \param[in] iter_max number of maximum iterations, scalar
    def set_iter_max(self, iter_max):
        self._iter_max = iter_max


    ## Get chosen value of maximum number of iterations for minimizer
    #  \return maximum number of iterations set for minimizer, scalar
    def get_iter_max(self):
        return self._iter_max


    def run_reconstruction(self):

        if self._reg_type in ["TK0"]:
            print("Chosen regularization type: zero-order Tikhonov")
            print("Regularization parameter = " + str(self._alpha))
            print("Maximum number of iterations = " + str(self._iter_max))
            # print("Tolerance = %.0e" %(self._tolerance))

            N_voxels = self._N_total_slice_voxels + self._N_voxels_HR_volume
        else:
            print("Chosen regularization type: first-order Tikhonov")
            print("Regularization parameter = " + str(self._alpha))
            print("Maximum number of iterations = " + str(self._iter_max))
            # print("Tolerance = %.0e" %(self._tolerance))

            N_voxels = self._N_total_slice_voxels + 3*self._N_voxels_HR_volume

        ## Construct right-hand side b
        b = self._get_b(N_voxels)

        ## Construct linear operator A
        A_fw = lambda x: self._A[self._reg_type](x, N_voxels, self._alpha)
        A_bw = lambda x: self._A_adj[self._reg_type](x, self._alpha)
        A = LinearOperator((N_voxels, self._N_voxels_HR_volume), matvec=A_fw, rmatvec=A_bw)

        # HR_nda = sitk.GetArrayFromImage(self._HR_volume.sitk)

        # res = lsq_linear(A, b, bounds=(0, np.inf), max_iter=self._iter_max, lsq_solver=None, lsmr_tol='auto', verbose=2)
        # res = lsq_linear(A, b, max_iter=self._iter_max, lsq_solver=None, lsmr_tol='auto', verbose=2)
        # res = nnls(A,b) #does not work with sparse linear operator

        res = lsqr(A, b, iter_lim=self._iter_max, show=True) #Works neatly (but does not allow bounds)

        HR_nda_vec = res[0]

        ## After reconstruction: Update member attribute
        self._HR_volume.itk = self._get_HR_itk_image_from_array_vec( HR_nda_vec )
        self._HR_volume.sitk = sitkh.convert_itk_to_sitk_image( self._HR_volume.itk )


    """
    TK0-Solver
    """ 
    def _A_TK0(self, HR_nda_vec, N_voxels, alpha):

        A_x = np.zeros(N_voxels)
        A_x[0:-self._N_voxels_HR_volume] = self._MAx(HR_nda_vec)
        A_x[-self._N_voxels_HR_volume:] = HR_nda_vec*np.sqrt(alpha)

        return A_x


    def _A_adj_TK0(self, slice_nda_vec, alpha):

        A_adj_y = self._A_adj_M_y(slice_nda_vec)
        A_adj_y = A_adj_y + slice_nda_vec[-self._N_voxels_HR_volume:]*np.sqrt(alpha)

        return A_adj_y


    """
    TK1-Solver
    """
    def _A_TK1(self, HR_nda_vec, N_voxels, alpha):

        A_x = np.zeros(N_voxels)
        A_x[0:-3*self._N_voxels_HR_volume] = self._MAx(HR_nda_vec)
        A_x[-3*self._N_voxels_HR_volume:] = self._D(HR_nda_vec)*np.sqrt(alpha)

        return A_x


    def _A_adj_TK1(self, slice_nda_vec, alpha):

        A_adj_y = self._A_adj_M_y(slice_nda_vec)
        A_adj_y = A_adj_y + self._D_adj(slice_nda_vec).flatten()*np.sqrt(alpha)

        return A_adj_y


    def _D(self, HR_nda_vec):
        
        N_voxels = self._N_voxels_HR_volume
        D_x = np.zeros(3*N_voxels)

        HR_nda = HR_nda_vec.reshape( self._HR_shape_nda )

        D_x[0:N_voxels] = self._differential_operations.Dx(HR_nda).flatten()
        D_x[N_voxels:2*N_voxels] = self._differential_operations.Dy(HR_nda).flatten()
        D_x[2*N_voxels:3*N_voxels] = self._differential_operations.Dz(HR_nda).flatten()

        return D_x


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


    """
    Used for both TK0- and TK1-solver
    """

    def _MAx(self, HR_nda_vec):

        x_itk = self._get_HR_itk_image_from_array_vec(HR_nda_vec)
        MA_x = np.zeros(self._N_total_slice_voxels)

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

                MA_x[i_min:i_max] = slice_nda.flatten()

                i_min = i_max

        return MA_x


    def _A_adj_M_y(self, slice_nda_vec):

        A_adj_M_y = np.zeros(self._N_voxels_HR_volume)

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

                A_adj_M_y = A_adj_M_y + tmp_nda

                i_min = i_max

        return A_adj_M_y


    ## Compute
    def _get_b(self, N_voxels):
        b = np.zeros(N_voxels)

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

                
            