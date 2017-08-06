##
# \file PrimalDualSolver.py
# \brief      Solve reconstruction problem A_k x = y_k for all slices k via
#             Primal-Dual solver.
#
# Implementation to get an approximate solution of the inverse problem
# \f$ y_k = A_k x
# \f$ for each slice
# \f$ y_k,\,k=1,\dots,K
# \f$ by using first-order primal-dual algorithms for convex problems as
# introduced in Chambolle, A. & Pock, T., 2011. A First-Order Primal-Dual
# Algorithm for Convex Problems with Applications to Imaging. Journal of
# Mathematical Imaging and Vision, 40(1), pp.120-145.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

# Import libraries
import sys
import itk
import SimpleITK as sitk
import numpy as np
import time
from datetime import timedelta

import numericalsolver.PrimalDualSolver as pd
import numericalsolver.LinearOperators as linop
import pythonhelper.SimpleITKHelper as sitkh
import pythonhelper.PythonHelper as ph
from numericalsolver.ProximalOperators import ProximalOperators as prox

# Import modules
import volumetricreconstruction.reconstruction.solver.TikhonovSolver as tk
from volumetricreconstruction.reconstruction.solver.Solver import Solver


# This class implements the framework to iteratively solve
#  \f$ \vec{y}_k = A_k \vec{x} \f$ for every slice \f$ \vec{y}_k,\,k=1,\dots,K \f$
#  via first-order primal dual algorithms.
#  TODO
class PrimalDualSolver(Solver):

    def __init__(self,
                 stacks,
                 reconstruction,
                 alpha=0.03,
                 alpha_cut=3,
                 iter_max=10,
                 minimizer="lsmr",
                 data_loss="linear",
                 huber_gamma=1.345,
                 deconvolution_mode="full_3D",
                 predefined_covariance=None,
                 reg_type="TV",
                 reg_huber_gamma=0.05,
                 iterations=10,
                 alg_type="ALG2",
                 verbose=0,
                 ):

        super(self.__class__, self).__init__(
            stacks=stacks,
            reconstruction=reconstruction,
            alpha=alpha,
            alpha_cut=alpha_cut,
            iter_max=iter_max,
            minimizer=minimizer,
            data_loss=data_loss,
            huber_gamma=huber_gamma,
            deconvolution_mode=deconvolution_mode,
            predefined_covariance=predefined_covariance,
            verbose=verbose,
        )

        # regularization type
        self._reg_type = reg_type

        # number of primal-dual iterations
        self._iterations = iterations

        # parameter used for Huber regularizer
        self._reg_huber_gamma = reg_huber_gamma

        # define method to update parameter
        self._alg_type = alg_type

    def get_setting_specific_filename(self, prefix="SRR_"):

        # Build filename
        filename = prefix
        filename += "stacks" + str(len(self._stacks))
        if self._alpha > 0:
            filename += "_PrimalDual"
            filename += "_" + self._reg_type
            if self._reg_type == "huber":
                filename += "_gamma" + str(self._reg_huber_gamma)
            # filename += "_" + self._alg_type
        filename += "_" + self._minimizer
        if self._data_loss not in ["linear"] or self._minimizer in ["L-BFGS-B"]:
            filename += "_" + self._data_loss
            if self._data_loss in ["huber"]:
                filename += str(self._huber_gamma)
        filename += "_alpha" + str(self._alpha)
        filename += "_itermax" + str(self._iter_max)
        filename += "_PDiterations" + str(self._iterations)

        # Replace dots by 'p'
        filename = filename.replace(".", "p")

        return filename

    def _run_reconstruction(self, verbose=0):

        if self._reg_type not in ["TV", "huber"]:
            raise ValueError("Error: regularization type can only be either "
                             "'TV' or 'huber'")

        self._print_info_text()

        # L^2 = ||K||^2 = ||\nabla||^2 = ||div||^2 <= 16/h^2 in 3D
        # However, it seems that the smaller L2 the bigger the effect of TV
        # regularization. Try, e.g. L2 = 1.
        L2 = 16. / self._reconstruction.sitk.GetSpacing()[0]**2

        # Get operators
        A = self.get_A()
        A_adj = self.get_A_adj()
        b = self.get_b()
        x0 = self.get_x0()
        x_scale = x0.max()

        spacing = np.array(self._reconstruction.sitk.GetSpacing())
        linear_operators = linop.LinearOperators3D(spacing=spacing)
        grad, grad_adj = linear_operators.get_gradient_operators()

        X_shape = self._reconstruction_shape
        Z_shape = grad(x0.reshape(*X_shape)).shape

        B = lambda x: grad(x.reshape(*X_shape)).flatten()
        B_adj = lambda x: grad_adj(x.reshape(*Z_shape)).flatten()

        prox_f = lambda x, tau: prox.prox_linear_least_squares(
            x=x, tau=tau,
            A=A, A_adj=A_adj,
            b=b, x0=x0, x_scale=x_scale,
            iter_max=self._iter_max,
            verbose=self._verbose)

        if self._reg_type == "TV":
            prox_g_conj = prox.prox_tv_conj
        elif self._reg_type == "huber":
            prox_g_conj = lambda x, sigma: prox.prox_huber_conj(
                x, sigma, self._reg_huber_gamma)

        solver = pd.PrimalDualSolver(
            prox_f=prox_f,
            prox_g_conj=prox_g_conj,
            B=B,
            B_conj=B_adj,
            L2=L2,
            x0=x0,
            x_scale=x_scale,
            alpha=self._alpha,
            iterations=self._iterations,
            verbose=self._verbose,
            alg_type=self._alg_type,
        )
        solver.run()

        # Get computational time
        self._computational_time = solver.get_computational_time()

        # Update volume
        self._reconstruction.itk = self._get_itk_image_from_array_vec(
            solver.get_x(), self._reconstruction.itk)
        self._reconstruction.sitk = sitkh.get_sitk_from_itk_image(
            self._reconstruction.itk)

    def _print_info_text(self):
        ph.print_title("Primal-Dual Solver:")
        ph.print_info("Chosen regularization type: %s" %
                      (self._reg_type), newline=False)
        if self._reg_type == "huber":
            print(" (gamma = %g)" % (self._reg_huber_gamma))
        else:
            print("")
        ph.print_info("Strategy for parameter update: %s"
                      % (self._alg_type))
        ph.print_info(
            "Regularization parameter alpha: %g" % (self._alpha))
        if self._data_loss in ["huber"]:
            ph.print_info("Loss function: %s (gamma = %g)" %
                          (self._data_loss, self._huber_gamma))
        else:
            ph.print_info("Loss function: %s" % (self._data_loss))
        ph.print_info("Number of Primal-Dual iterations: %d" %
                      (self._iterations))
        ph.print_info("Minimizer: %s" % (self._minimizer))
        ph.print_info(
            "Maximum number of iterations: %d" % (self._iter_max))
