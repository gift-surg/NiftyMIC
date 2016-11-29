#!/usr/bin/python

## \file NonnegativeTikhonovSolver.py
#  \brief Implementation to get an approximate solution of the inverse problem 
#  \f$ y_k = A_k x \f$ for each slice \f$ y_k,\,k=1,\dots,K \f$
#  by using Tikhonov-regularization with non-negative constraints.
#  Solution via Alternating Direction Method of Multipliers (ADMM) method.
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date Aug 2016

## Import libraries
import os                       # used to execute terminal commands in python
import sys
import itk
import SimpleITK as sitk
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import lsmr
from scipy.optimize import lsq_linear
from scipy.optimize import nnls
import time
from datetime import timedelta

## Import modules
import utilities.SimpleITKHelper as sitkh
from reconstruction.solver.TikhonovSolver import TikhonovSolver


## This class implements the framework to iteratively solve 
#  \f$ \vec{y}_k = A_k \vec{x} \f$ for every slice \f$ \vec{y}_k,\,k=1,\dots,K \f$
#  with non-negative constraints by using Tikhonov-regularization and ADMM.
#  TODO
class NonnegativeTikhonovSolver(TikhonovSolver):

    ##
    #          Constructor
    # \date          2016-08-01 22:57:21+0100
    #
    # \param         self                                    The object
    # \param[in]     stacks                                  list of Stack
    #                                                        objects containing
    #                                                        all stacks used
    #                                                        for the
    #                                                        reconstruction
    # \param[in,out] HR_volume                               Stack object
    #                                                        containing the
    #                                                        current estimate
    #                                                        of the HR volume
    #                                                        (used as initial
    #                                                        value + space
    #                                                        definition)
    # \param[in]     alpha_cut                               Cut-off distance
    #                                                        for Gaussian
    #                                                        blurring filter
    # \param[in]     alpha                                   regularization
    #                                                        parameter, scalar
    # \param[in]     iter_max                                number of maximum
    #                                                        iterations, scalar
    # \param[in]     rho                                     regularization
    #                                                        parameter of
    #                                                        augmented
    #                                                        Lagrangian term,
    #                                                        scalar
    # \param[in]     ADMM_iterations                         number of ADMM
    #                                                        iterations, scalar
    # \param[in]     ADMM_iterations_output_dir              The ADMM iterations output dir
    # \param[in]     ADMM_iterations_output_filename_prefix  The ADMM iterations output filename prefix
    #
    def __init__(self, stacks, HR_volume, alpha_cut=3, alpha=0.03, iter_max=10, reg_type="TK1", rho=0.5, ADMM_iterations=10, ADMM_iterations_output_dir=None, ADMM_iterations_output_filename_prefix="TV-L2"):

        self._MINIMIZER = "lsmr"
        # self._MINIMIZER = "lsqr"

        ## Run constructor of superclass
        TikhonovSolver.__init__(self, stacks, HR_volume, alpha_cut, alpha, iter_max, reg_type, self._MINIMIZER)               
        
        ## Settings for optimizer
        self._rho = rho
        self._ADMM_iterations = ADMM_iterations

        self._ADMM_iterations_output_dir = ADMM_iterations_output_dir
        self._ADMM_iterations_output_filename_prefix = ADMM_iterations_output_filename_prefix


    ## Set regularization parameter used for augmented Lagrangian in TV-L2 regularization
    #  \[$
    #   \sum_{k=1}^K \frac{1}{2} \Vert y_k - A_k x \Vert_{\ell^2}^2 + \alpha\,\Psi(x) 
    #   + \mu \cdot (\nabla x - v) + \frac{\rho}{2} \Vert \nabla x - v \Vert_{\ell^2}^2
    #  \]$
    #  \param[in] rho regularization parameter of augmented Lagrangian term, scalar
    def set_rho(self, rho):
        self._rho = rho


    ## Get regularization parameter used for augmented Lagrangian in TV-L2 regularization
    #  \return regularization parameter of augmented Lagrangian term, scalar
    def get_rho(self):
        return self._rho


    ## Set ADMM iterations to solve TV-L2 reconstruction problem
    #  \[$
    #   \sum_{k=1}^K \frac{1}{2} \Vert y_k - A_k x \Vert_{\ell^2}^2 + \alpha\,\Psi(x) 
    #   + \mu \cdot (\nabla x - v) + \frac{\rho}{2} \Vert \nabla x - v \Vert_{\ell^2}^2
    #  \]$
    #  \param[in] iterations number of ADMM iterations, scalar
    def set_ADMM_iterations(self, iterations):
        self._ADMM_iterations = iterations


    ## Get chosen value of ADMM iterations to solve TV-L2 reconstruction problem
    #  \return number of ADMM iterations, scalar
    def get_ADMM_iterations(self):
        return self._ADMM_iterations


    ## Set ouput directory to write TV results in case outputs of ADMM iterations are desired
    #  \param[in] dir_output directory to write TV results, string
    def set_ADMM_iterations_output_dir(self, dir_output):
        self._ADMM_iterations_output_dir = dir_output


    ## Get ouput directory to write TV results in case outputs of ADMM iterations are desired
    def get_ADMM_iterations_output_dir(self):
        return self._ADMM_iterations_output_dir


    ## Set filename prefix to write TV reconstructed volumes of ADMM iteration results 
    #  \pre ADMM_iterations_output_dir was set
    #  \param[in] filename filename prefix of ADMM output iteration volumes, string
    def set_ADMM_iterations_output_filename_prefix(self, filename):
        self._ADMM_iterations_output_filename_prefix = filename


    ## Get filename to write TV reconstructed volumes of ADMM iteration results 
    #  \pre ADMM_iterations_output_dir was set
    def get_ADMM_iterations_output_filename_prefix(self, filename):
        return self._ADMM_iterations_output_filename_prefix


    ##
    #       Run the reconstruction algorithm based on Tikhonov
    #             regularization
    # \post       self._HR_volume is updated with new volume and can be fetched
    #             by \p get_HR_volume
    # \date       2016-08-04 16:19:40+0100
    #
    # \param      self                   The object
    # \param[in]  provide_initial_value  Use HR volume during initialization as
    #                                    initial value, boolean. Otherwise,
    #                                    assume zero initial vale.
    #
    def run_reconstruction(self, provide_initial_value=True):
        
        ## Compute number of voxels to be stored for augmented linear system
        if self._reg_type in ["TK0"]:
            ## G = Identity:
            N_voxels = self._N_total_slice_voxels + 2*self._N_voxels_HR_volume
            
            print("Chosen regularization type: zero-order Tikhonov with non-negativity constraints")

        else:
            ## G = [Dx, Dy, Dz]^T, i.e. gradient computation:
            N_voxels = self._N_total_slice_voxels + 4*self._N_voxels_HR_volume

            print("Chosen regularization type: first-order Tikhonov with non-negativity constraints")

        print("Minimizer: " + self._MINIMIZER)
        print("Regularization parameter: " + str(self._alpha))
        print("Maximum number of TK solver iterations: " + str(self._iter_max))
        print("Regularization parameter of augmented Lagrangian term rho: " + str(self._rho))
        print("Number of ADMM iterations: " + str(self._ADMM_iterations))
        # print("Tolerance: %.0e" %(self._tolerance))

        time_start = time.time()

        ## Construct (sparse) linear operator A
        A_fw = lambda x: np.concatenate((self._A[self._reg_type](x, self._alpha), np.sqrt(self._rho)*x))
        A_bw = lambda y: self._A_adj[self._reg_type](y, self._alpha) + np.sqrt(self._rho)*y[-self._N_voxels_HR_volume:]
        A = LinearOperator((N_voxels, self._N_voxels_HR_volume), matvec=A_fw, rmatvec=A_bw)

        ## Construct right-hand side b
        b = self._get_b(N_voxels)

        HR_nda_vec = sitk.GetArrayFromImage(self._HR_volume.sitk).flatten()
        # v = np.zeros_like(HR_nda_vec)
        # w = np.zeros_like(HR_nda_vec)
        v = np.clip(HR_nda_vec, 0, np.inf)
        w = HR_nda_vec - v
        

        for i_ADMM in range(0, self._ADMM_iterations):

            ## Update RHS
            b[-self._N_voxels_HR_volume:] = np.sqrt(self._rho)*(v-w)

            ## Incorporate initial value for least-squares solver:
            b_centered = b - A_fw(HR_nda_vec)          

            ## Linear least-squares solver: 
            delta_HR_nda_vec = lsmr(A, b_centered, maxiter=self._iter_max, show=False)[0]

            ## Correct for shift
            HR_nda_vec += delta_HR_nda_vec

            ## Compute auxilary variables
            v = np.clip(HR_nda_vec, 0, np.inf)
            w = w + HR_nda_vec - v

            print("\tADMM-iteration " + str(i_ADMM+1) + ": norm(x-v) = " + str(np.linalg.norm(HR_nda_vec-v)) + ", min(x) = " + str(np.min(HR_nda_vec)))

            recon_step_itk = self._get_itk_image_from_array_vec( HR_nda_vec, self._HR_volume.itk )
            # sitkh.show_itk_image(recon_step_itk,title="ADMM_iteration_"+str(i_ADMM+1))
            sitkh.write_itk_image(recon_step_itk,"/tmp/"+"ADMM_iteration_"+str(i_ADMM+1)+".nii.gz")


        ## Set elapsed time
        time_end = time.time()
        self._elapsed_time_sec = time_end-time_start

        ## After reconstruction: Update member attribute
        self._HR_volume.itk = self._get_itk_image_from_array_vec( HR_nda_vec, self._HR_volume.itk )
        self._HR_volume.sitk = sitkh.get_sitk_from_itk_image( self._HR_volume.itk )
    