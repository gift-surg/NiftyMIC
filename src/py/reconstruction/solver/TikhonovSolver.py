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
import time
from scipy.optimize import minimize

## Add directories to import modules
dir_src_root = "../src/"
sys.path.append( dir_src_root )

## Import modules
import utilities.SimpleITKHelper as sitkh
from reconstruction.solver.Solver import Solver


## This class implements the framework to iteratively solve 
#  \f$ \vec{y}_k = A_k \vec{x} \f$ for every slice \f$ \vec{y}_k,\,k=1,\dots,K \f$
#  via Tikhonov-regularization via an augmented least-square approach
#  where \f$A_k=D_k B_k W_k\in\mathbb{R}^{N_k}\f$ denotes the combined warping, blurring and downsampling 
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
#           - \begin{pmatrix} M_1 \vec{y}_1 \\ M_2 \vec{y}_2 \\ \vdots \\ M_K \vec{y}_K \\ \vec{0} \end{pmatrix}
#       \Bigg\Vert_{\ell^2}^2 \Bigg)
#  \f] 
#  By defining the shorthand 
#  \f[ 
#   MA := \begin{pmatrix} M_1 A_1 \\ M_2 A_2 \\ \vdots \\ M_K A_K \end{pmatrix}\in\mathbb{R}^{\sum_k N_k} \quad\text{and}\quad
#   M\vec{y} := \begin{pmatrix} M_1 \vec{y}_1 \\ M_2 \vec{y}_2 \\ \vdots \\ M_K \vec{y}_K \end{pmatrix}\in\mathbb{R}^{\sum_k N_k} 
#  \f]
#  the problem can be compactly written as
#  \f[
#       \text{arg min}_{\vec{x}} \Bigg( \Bigg\Vert 
#           \begin{pmatrix} MA \\ \sqrt{\alpha} G \end{pmatrix} \vec{x}
#           - \begin{pmatrix} M\vec{y} \\ \vec{0} \end{pmatrix}
#       \Bigg\Vert_{\ell^2}^2 \Bigg)
#  \f]
#  with \f$ G\in\mathbb{R}^N \f$ in case of \f$G=I\f$ or 
#  \f$G\in\mathbb{R}^{3N}\f$ in case of \f$G\f$ representing the gradient.
#  \see \p itkAdjointOrientedGaussianInterpolateImageFilter of \p ITK
#  \see \p itOrientedGaussianInterpolateImageFunction of \p ITK
class TikhonovSolver(Solver):

    ##
    #          Constructor
    # \date          2016-08-01 23:00:04+0100
    #
    # \param         self                The object
    # \param[in]     stacks              list of Stack objects containing all
    #                                    stacks used for the reconstruction
    # \param[in,out] HR_volume           Stack object containing the current
    #                                    estimate of the HR volume (used as
    #                                    initial value + space definition)
    # \param[in]     alpha_cut           Cut-off distance for Gaussian blurring
    #                                    filter
    # \param[in]     alpha               regularization parameter, scalar
    # \param[in]     iter_max            number of maximum iterations, scalar
    # \param[in]     reg_type            Type of Tikhonov regualrization, i.e.
    #                                    TK0 or TK1 for either zeroth- or first
    #                                    order Tikhonov
    # \param[in]     minimizer           Type of minimizer used to solve
    #                                    minimization problem, possible types:
    #                                    'lsmr', 'lsqr', 'L-BFGS-B' #
    # \param[in]     deconvolution_mode  Either "full_3D" or "only_in_plane".
    #                                    Indicates whether full 3D or only
    #                                    in-plane deconvolution is considered
    #
    def __init__(self, stacks, HR_volume, alpha_cut=3, alpha=0.03, iter_max=10, reg_type="TK1", minimizer="lsmr", deconvolution_mode="full_3D", loss="linear", huber_gamma=1.345, predefined_covariance=None):

        ## Run constructor of superclass
        Solver.__init__(self, stacks=stacks, HR_volume=HR_volume, alpha_cut=alpha_cut, alpha=alpha, iter_max=iter_max, minimizer=minimizer, deconvolution_mode=deconvolution_mode, loss=loss, huber_gamma=huber_gamma, predefined_covariance=predefined_covariance)
        
        ## Settings for optimizer
        self._reg_type = reg_type

        self._A = {
            "TK0"   : self._A_TK0,
            "TK1"   : self._A_TK1
        }

        self._A_adj = {
            "TK0"   : self._A_adj_TK0,
            "TK1"   : self._A_adj_TK1
        }

        self._get_residual_prior = {
            "TK0"   : self._get_residual_prior_TK0,
            "TK1"   : self._get_residual_prior_TK1
        }
        self._get_gradient_residual_prior = {
            "TK0"   : self._get_gradient_residual_prior_TK0,
            "TK1"   : self._get_gradient_residual_prior_TK1
        }

        ## Residual values after optimization
        self._residual_prior = None
        self._residual_ell2 = None


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


    ## Compute statistics associated to performed reconstruction
    def compute_statistics(self):
        HR_nda_vec = sitk.GetArrayFromImage(self._HR_volume.sitk).flatten()

        self._residual_ell2 = self._get_residual_ell2(HR_nda_vec)
        self._residual_prior = self._get_residual_prior[self._reg_type](HR_nda_vec)

    ##
    # Gets the final cost after optimization
    # \date       2016-11-25 18:33:00+0000
    #
    # \param      self  The object
    #
    # \return     The final cost.
    #
    def get_final_cost(self):

        if self._residual_ell2 is None or self._residual_prior is None:
            self.compute_statistics()

        return self._residual_ell2 + self._alpha*self._residual_prior
    

    ##
    #       Print statistics associated to performed reconstruction
    # \date       2016-07-29 12:30:30+0100
    #
    # \param      self  The object
    #
    def print_statistics(self):
        print("\nStatistics for performed reconstruction with %s-regularization:" %(self._reg_type))
        # if self._elapsed_time_sec < 0:
        #     raise ValueError("Error: Elapsed time has not been measured. Run 'run_reconstruction' first.")
        # else:
        print("\tElapsed time: %s" %(self.get_computational_time()))
        if self._residual_ell2 is not None:
            print("\tell^2-residual sum_k ||M_k(A_k x - y_k)||_2^2 = %.3e" %(self._residual_ell2))
            print("\tprior residual = %.3e" %(self._residual_prior))
        else:
            print("\tRun 'compute_statistics' for data and prior residuals")


    ##
    #       Gets the setting specific filename indicating the information
    #             used for the reconstruction step
    # \date       2016-11-17 15:41:58+0000
    #
    # \param      self    The object
    # \param      prefix  The prefix as string
    #
    # \return     The setting specific filename as string.
    #
    def get_setting_specific_filename(self, prefix="SRR_"):
        
        ## Build filename
        filename = prefix
        filename += "stacks" + str(len(self._stacks))
        if self._alpha > 0:
            filename += "_" + self._reg_type
        filename += "_" + self._minimizer
        if self._loss not in ["linear"] or self._minimizer in ["L-BFGS-B"]:
            filename += "_" + self._loss
            if self._loss in ["huber"]:
                filename += str(self._huber_gamma)
        filename += "_alpha" + str(self._alpha)
        filename += "_itermax" + str(self._iter_max)

        ## Replace dots by 'p'
        filename = filename.replace(".","p")

        return filename


    ##
    #       Run the reconstruction algorithm based on Tikhonov
    #             regularization
    # \date       2016-07-29 12:35:01+0100
    # \post       self._HR_volume is updated with new volume and can be fetched
    #             by \p get_HR_volume
    #
    # \param      self                   The object
    # \param[in]  provide_initial_value  Use HR volume during initialization as
    #                                    initial value, boolean. Otherwise, 
    #                                    assume zero initial vale.
    #
    def run_reconstruction(self):

        ## Compute number of voxels to be stored for augmented linear system
        if self._reg_type in ["TK0"]:            
            print("Chosen regularization type: zeroth-order Tikhonov")

        else:
            print("Chosen regularization type: first-order Tikhonov")

        if self._deconvolution_mode in ["only_in_plane"]:
            print("(Only in-plane deconvolution is performed)")

        elif self._deconvolution_mode in ["predefined_covariance"]:
            print("(Predefined covariance used: cov = diag(%s))" % (np.diag(self._predefined_covariance)))

        sys.stdout.write("Loss function: %s" %(self._loss))
        if self._loss in ["huber"]:
            print(" (gamma = %g)" %(self._huber_gamma))

        print("Regularization parameter: " + str(self._alpha))

        ## Non-linear loss function requires use of L-BFGS-B
        if self._loss not in ["linear"] and self._minimizer not in ["L-BFGS-B"]:
            print("Note, selected minimizer '%s' cannot be used. Non-linear loss function requires L-BFGS-B." %(self._minimizer))
            self._minimizer = "L-BFGS-B"

        print("Minimizer: " + self._minimizer)
        print("Maximum number of iterations: " + str(self._iter_max))
        # print("Tolerance: %.0e" %(self._tolerance))

        time_start = time.time()

        ## Remark: L-BFGS-B is faster with direct computation as done in
        # _run_reconstruction_nonlinear. Thus, _run_reconstruction_nonlinear
        # preferred in linear case for L-BFGS-B too.
        if self._loss not in ["linear"] or self._minimizer in ["L-BFGS-B"]:
            HR_nda_vec = self._run_reconstruction_nonlinear()
        else:
            HR_nda_vec = self._run_reconstruction_linear()

        ## Set elapsed time
        self._elapsed_time_sec = time.time()-time_start

        ## After reconstruction: Update member attribute
        self._HR_volume.itk = self._get_itk_image_from_array_vec( HR_nda_vec, self._HR_volume.itk )
        self._HR_volume.sitk = sitkh.get_sitk_from_itk_image( self._HR_volume.itk )


    def _run_reconstruction_linear(self):

        ## Construct (sparse) linear operator A
        A_fw = lambda x: self._A[self._reg_type](x, self._alpha)
        A_bw = lambda x: self._A_adj[self._reg_type](x, self._alpha)

        ## Construct right-hand side b
        b = self._get_b()

        ## Run solver
        HR_nda_vec = self._get_approximate_solution[self._minimizer](A_fw, A_bw, b)

        return HR_nda_vec


    def _run_reconstruction_nonlinear(self, x0=None):

        ## Set initial value and bounds
        if x0 is None:
            # x0 = np.clip(sitk.GetArrayFromImage(self._HR_volume.sitk).flatten(), 0, np.inf)
            x0 = np.zeros(np.array(self._HR_volume.sitk.GetSize())[::-1])
        else:
            ## In case initial value is given, the non-masked voxels will
            ## smoothly vary but will not be zero! Hence, prefer zero-init
            x0 = np.clip(x0, 0, np.inf)
        bounds = [[0,None]]*x0.size

        ## Construct right-hand side b
        b = self._get_M_y()

        # if self._loss in ["huber"]:
        #     self._get_loss[self._loss] = lambda x : self._get_loss[self._loss](x, self._huber_gamma)
        #     self._get_gradient_loss[self._loss] = lambda x : self._get_gradient_loss[self._loss](x, self._huber_gamma)

        ## Set cost function and its jacobian
        fun = lambda x: self._get_fun_nonlinear(x, b, self._alpha)
        jac = lambda x: self._get_jac_nonlinear(x, b, self._alpha)

        ## Run solver
        HR_nda_vec = minimize(method='L-BFGS-B', fun=fun, x0=x0, options={'maxiter': self._iter_max, 'disp': True}, jac=jac, bounds=bounds).x

        return HR_nda_vec


    def _get_fun_nonlinear(self, x, b, alpha):

        residual = self._MA(x) - b

        data_term = 0.5*np.sum( self._get_loss[self._loss](residual**2) )

        regularizer_term = 0.5*alpha*self._get_residual_prior[self._reg_type](x)

        return data_term + regularizer_term


    def _get_jac_nonlinear(self, x, b, alpha):

        residual = self._MA(x) - b

        grad_data_term = self._A_adj_M(self._get_gradient_loss[self._loss](residual**2) * residual)
        grad_regularizer_term = alpha * self._get_gradient_residual_prior[self._reg_type](x)

        return grad_data_term + grad_regularizer_term


    ## Compute
    #  \f$ b := \begin{pmatrix} M_1 \vec{y}_1 \\ M_2 \vec{y}_2 \\ \vdots \\ M_K \vec{y}_K \\ \vec{0}\end{pmatrix} \f$ 
    #  \param[in] N_voxels number of voxels (only two possibilities depending on G), integer
    #  \return vector b as 1D array
    def _get_b(self):

        ## Compute number of voxels to be stored for augmented linear system
        if self._reg_type in ["TK0"]:
            ## G = Identity:
            N_voxels = self._N_total_slice_voxels + self._N_voxels_HR_volume
            
        else:
            ## G = [Dx, Dy, Dz]^T, i.e. gradient computation:
            N_voxels = self._N_total_slice_voxels + 3*self._N_voxels_HR_volume

        ## Allocate memory
        b = np.zeros(N_voxels)

        ## Compute M y, i.e. masked slices stacked to 1D vector
        b[0:self._N_total_slice_voxels] = self._get_M_y()

        return b


    ## Evaluate augmented linear operator for TK0-regularization, i.e.
    #  \f$
    #       \begin{pmatrix} MA \\ \sqrt{\alpha} G \end{pmatrix} \vec{x}
    #     = \begin{pmatrix} M_1 A_1 \\ M_2 A_2 \\ \vdots \\ M_K A_K \\ \sqrt{\alpha} I \end{pmatrix} \vec{x}
    #  \f$
    #  for \f$ G = I\f$ identity matrix
    #  \param[in] HR_nda_vec HR data as 1D array
    #  \param[in] alpha regularization parameter, scalar
    #  \return evaluated augmented linear operator as 1D array
    def _A_TK0(self, HR_nda_vec, alpha):

        ## Allocate memory
        A_x = np.zeros(self._N_total_slice_voxels+self._N_voxels_HR_volume)

        ## Compute MA x
        A_x[0:-self._N_voxels_HR_volume] = self._MA(HR_nda_vec)

        ## Compute sqrt(alpha)*x
        A_x[-self._N_voxels_HR_volume:] = np.sqrt(alpha)*HR_nda_vec

        return A_x


    ## Evaluate the adjoint augmented linear operator for TK0-regularization, i.e.
    #  \f$
    #       \begin{bmatrix} A^* M && \sqrt{\alpha} G^* \end{bmatrix} \vec{y}
    #     = \begin{bmatrix} A_1^* M_1 && A_2^* M_2 && \cdots && A_K^* M_K && \sqrt{\alpha} I \end{bmatrix} \vec{y}
    #  \f$
    #  for \f$ G = I\f$ identity matrix and \f$\vec{y}\in\mathbb{R}^{\sum_k N_k + N}\f$ 
    #  representing a vector of stacked slices
    #  \param[in] stacked_slices_nda_vec stacked slice data as 1D array
    #  \param[in] alpha regularization parameter, scalar
    #  \return evaluated augmented adjoint linear operator as 1D array
    def _A_adj_TK0(self, stacked_slices_nda_vec, alpha):

        ## Compute A'M y[upper] 
        A_adj_y = self._A_adj_M(stacked_slices_nda_vec)

        ## Add sqrt(alpha)*y[lower]
        A_adj_y = A_adj_y + stacked_slices_nda_vec[-self._N_voxels_HR_volume:]*np.sqrt(alpha)

        return A_adj_y

    
    ## Compute residual for TK1-regularization prior, i.e. 
    #  || x ||^2 
    #  \param[in] HR_nda_vec HR data as 1D array
    #  \return || x ||^2 
    def _get_residual_prior_TK0(self, HR_nda_vec):
        return np.sum( HR_nda_vec**2 )


    ##
    # Gets the gradient of TK0-prior.
    # \date       2017-05-15 19:43:32+0100
    #
    # \param      self        The object
    # \param      HR_nda_vec  The hr nda vector
    #
    # \return     The gradient residual prior tk 0.
    #
    def _get_gradient_residual_prior_TK0(self, HR_nda_vec):
        return HR_nda_vec


    ## Compute residual for TK1-regularization prior, i.e. 
    #  || Dx ||^2 with D = [D_x; D_y; D_z]
    #  \param[in] HR_nda_vec HR data as 1D array
    #  \return || Dx ||^2 
    def _get_residual_prior_TK1(self, HR_nda_vec):
        HR_nda = HR_nda_vec.reshape(self._HR_shape_nda)

        Dx = self._differential_operations.Dx(HR_nda)
        Dy = self._differential_operations.Dy(HR_nda)
        Dz = self._differential_operations.Dz(HR_nda)

        ## Compute norm || Dx ||^2 with D = [D_x; D_y; D_z]
        return np.sum( Dx**2 ) + np.sum( Dy**2 ) + np.sum( Dz**2 )

    
    ##
    # Gets the gradient of TK1-prior.
    # \date       2017-05-15 19:43:52+0100
    #
    # \param      self        The object
    # \param      HR_nda_vec  The hr nda vector
    #
    # \return     The gradient residual prior tk 1.
    #
    def _get_gradient_residual_prior_TK1(self, HR_nda_vec):
        return self._D_adj(self._D(HR_nda_vec)).flatten()       


