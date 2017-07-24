#!/usr/bin/python

# \file TikhonovSolver.py
#  \brief Implementation to get an approximate solution of the inverse problem
#  \f$ y_k = A_k x \f$ for each slice \f$ y_k,\,k=1,\dots,K \f$
#  by using Tikhonov-regularization
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date July 2016

# Import libraries
import os                       # used to execute terminal commands in python
import sys
import itk
import SimpleITK as sitk
import numpy as np
import time
from scipy.optimize import minimize

import numericalsolver.TikhonovLinearSolver as tk
import numericalsolver.LinearOperators as linop
import pythonhelper.SimpleITKHelper as sitkh

# Import modules
from reconstruction.solver.Solver import Solver


# This class implements the framework to iteratively solve
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
    # Constructor
    # \date          2016-08-01 23:00:04+0100
    #
    # \param         self                   The object
    # \param[in]     stacks                 list of Stack objects containing
    #                                       all stacks used for the
    #                                       reconstruction
    # \param[in,out] HR_volume              Stack object containing the current
    #                                       estimate of the HR volume (used as
    #                                       initial value + space definition)
    # \param[in]     alpha_cut              Cut-off distance for Gaussian
    #                                       blurring filter
    # \param[in]     alpha                  regularization parameter, scalar
    # \param[in]     iter_max               number of maximum iterations,
    #                                       scalar
    # \param[in]     reg_type               Type of Tikhonov regualrization,
    #                                       i.e. TK0 or TK1 for either zeroth-
    #                                       or first order Tikhonov
    # \param[in]     minimizer              Type of minimizer used to solve
    #                                       minimization problem, possible
    #                                       types: 'lsmr', 'lsqr', 'L-BFGS-B' #
    # \param[in]     deconvolution_mode     Either "full_3D" or
    #                                       "only_in_plane". Indicates whether
    #                                       full 3D or only in-plane
    #                                       deconvolution is considered
    # \param         loss                   The loss
    # \param         huber_gamma            The huber gamma
    # \param         predefined_covariance  The predefined covariance
    #
    def __init__(self,
                 stacks,
                 HR_volume,
                 alpha_cut=3,
                 alpha=0.03,
                 iter_max=10,
                 reg_type="TK1",
                 minimizer="lsmr",
                 deconvolution_mode="full_3D",
                 data_loss="linear",
                 huber_gamma=1.345,
                 predefined_covariance=None,
                 verbose=1,
                 ):

        # Run constructor of superclass
        Solver.__init__(self,
                        stacks=stacks,
                        HR_volume=HR_volume,
                        alpha_cut=alpha_cut,
                        alpha=alpha,
                        iter_max=iter_max,
                        minimizer=minimizer,
                        deconvolution_mode=deconvolution_mode,
                        data_loss=data_loss,
                        huber_gamma=huber_gamma,
                        predefined_covariance=predefined_covariance,
                        verbose=verbose)

        # Settings for optimizer
        self._reg_type = reg_type

        # Residual values after optimization
        self._residual_prior = None
        self._residual_ell2 = None

    # Set type of regularization. It can be either 'TK0' or 'TK1'
    #  \param[in] reg_type Either 'TK0' or 'TK1', string
    def set_regularization_type(self, reg_type):
        self._reg_type = reg_type

    # Get chosen type of regularization.
    #  \return regularization type as string
    def get_regularization_type(self):
        return self._reg_type

    # Compute statistics associated to performed reconstruction
    def compute_statistics(self):
        HR_nda_vec = sitk.GetArrayFromImage(self._HR_volume.sitk).flatten()

        self._residual_ell2 = self._get_residual_ell2(HR_nda_vec)
        self._residual_prior = self._get_residual_prior[
            self._reg_type](HR_nda_vec)

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
        print("\nStatistics for performed reconstruction with %s-regularization:" %
              (self._reg_type))
        # if self._elapsed_time_sec < 0:
        #     raise ValueError("Error: Elapsed time has not been measured. Run 'run_reconstruction' first.")
        # else:
        print("\tElapsed time: %s" % (self.get_computational_time()))
        if self._residual_ell2 is not None:
            print("\tell^2-residual sum_k ||M_k(A_k x - y_k)||_2^2 = %.3e" %
                  (self._residual_ell2))
            print("\tprior residual = %.3e" % (self._residual_prior))
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

        # Build filename
        filename = prefix
        filename += "stacks" + str(len(self._stacks))
        if self._alpha > 0:
            filename += "_" + self._reg_type
        filename += "_" + self._minimizer
        if self._data_loss not in ["linear"] or self._minimizer in ["L-BFGS-B"]:
            filename += "_" + self._data_loss
            if self._data_loss in ["huber"]:
                filename += str(self._huber_gamma)
        filename += "_alpha" + str(self._alpha)
        filename += "_itermax" + str(self._iter_max)

        # Replace dots by 'p'
        filename = filename.replace(".", "p")

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

        if self._reg_type not in ["TK0", "TK1"]:
            raise ValueError(
                "Error: regularization type can only be either 'TK0' or 'TK1'")

        # Compute number of voxels to be stored for augmented linear system
        if self._reg_type in ["TK0"]:
            print("Chosen regularization type: zeroth-order Tikhonov")

        else:
            print("Chosen regularization type: first-order Tikhonov")

        if self._deconvolution_mode in ["only_in_plane"]:
            print("(Only in-plane deconvolution is performed)")

        elif self._deconvolution_mode in ["predefined_covariance"]:
            print("(Predefined covariance used: cov = diag(%s))" %
                  (np.diag(self._predefined_covariance)))

        if self._data_loss in ["huber"]:
            print("Loss function: %s (gamma = %g)" %
                  (self._data_loss, self._huber_gamma))
        else:
            print("Loss function: %s" % (self._data_loss))

        print("Regularization parameter: " + str(self._alpha))

        # Non-linear loss function requires use of L-BFGS-B
        if self._data_loss not in ["linear"] and self._minimizer not in ["L-BFGS-B"]:
            print("Note, selected minimizer '%s' cannot be used. Non-linear loss function requires L-BFGS-B." %
                  (self._minimizer))
            self._minimizer = "L-BFGS-B"

        print("Minimizer: " + self._minimizer)
        print("Maximum number of iterations: " + str(self._iter_max))
        # print("Tolerance: %.0e" %(self._tolerance))

        # Get operators
        A = self._get_A()
        A_adj = self._get_A_adj()
        b = self._get_b()
        x0 = self._get_x0()

        if self._reg_type == "TK0":
            B = lambda x: x.flatten()
            B_adj = lambda x: x.flatten()

        elif self._reg_type == "TK1":
            spacing = np.array(self._HR_volume.sitk.GetSpacing())
            linear_operators = linop.LinearOperators3D(spacing=spacing)
            grad, grad_adj = linear_operators.get_gradient_operators()

            X_shape = self._HR_shape_nda
            Z_shape = grad(x0.reshape(*X_shape)).shape

            B = lambda x: grad(x.reshape(*X_shape)).flatten()
            B_adj = lambda x: grad_adj(x.reshape(*Z_shape)).flatten()

        # Run reconstruction
        solver = tk.TikhonovLinearSolver(
            A=A,
            A_adj=A_adj,
            B=B,
            B_adj=B_adj,
            b=b,
            x0=x0,
            alpha=self._alpha,
            verbose=self._verbose,
        )
        solver.run()

        # Get computational time
        self._computational_time = solver.get_computational_time()

        # After reconstruction: Update member attribute
        self._HR_volume.itk = self._get_itk_image_from_array_vec(
            solver.get_x(), self._HR_volume.itk)
        self._HR_volume.sitk = sitkh.get_sitk_from_itk_image(
            self._HR_volume.itk)
