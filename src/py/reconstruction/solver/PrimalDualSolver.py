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

# Import modules
import utilities.SimpleITKHelper as sitkh
import utilities.PythonHelper as ph
import reconstruction.solver.TikhonovSolver as tk
from reconstruction.solver.Solver import Solver


# This class implements the framework to iteratively solve
#  \f$ \vec{y}_k = A_k \vec{x} \f$ for every slice \f$ \vec{y}_k,\,k=1,\dots,K \f$
#  via first-order primal dual algorithms.
#  TODO
class PrimalDualSolver(Solver):

    def __init__(self,
                 stacks,
                 HR_volume,
                 alpha=0.03,
                 alpha_cut=3,
                 iter_max=10,
                 minimizer="lsmr",
                 loss="linear",
                 huber_gamma=1.345,
                 deconvolution_mode="full_3D",
                 predefined_covariance=None,
                 reg_type="TVL2",
                 primal_dual_iterations=10,
                 ):

        super(self.__class__, self).__init__(
            stacks=stacks,
            HR_volume=HR_volume,
            alpha=alpha,
            alpha_cut=alpha_cut,
            iter_max=iter_max,
            minimizer=minimizer,
            loss=loss,
            huber_gamma=huber_gamma,
            deconvolution_mode=deconvolution_mode,
            predefined_covariance=predefined_covariance,
        )

        self._reg_type = reg_type
        self._primal_dual_iterations = primal_dual_iterations
        self._lambda = 1./alpha

    def get_setting_specific_filename(self, prefix="SRR_"):

        # Build filename
        filename = prefix
        filename += "stacks" + str(len(self._stacks))
        if self._alpha > 0:
            filename += "_PrimalDual_" + self._reg_type
        filename += "_" + self._minimizer
        if self._loss not in ["linear"] or self._minimizer in ["L-BFGS-B"]:
            filename += "_" + self._loss
            if self._loss in ["huber"]:
                filename += str(self._huber_gamma)
        filename += "_alpha" + str(self._alpha)
        filename += "_itermax" + str(self._iter_max)
        filename += "_PrimalDualIterations" + str(self._primal_dual_iterations)

        # Replace dots by 'p'
        filename = filename.replace(".", "p")

        return filename

    def print_statistics(self):
        ph.print_subtitle("Statistics")
        ph.print_debug_info("Elapsed time: %s" %
                            (self.get_computational_time()))

    def run_reconstruction(self, debug=0):

        self._print_info_text()

        time_start = time.time()

        # Not sure about L = ||K|| = ||\nabla||
        # (The smaller L2 the bigger the effect of TVL2 regularization)
        # L2 = 100
        # L2 = 8. / self._HR_volume.sitk.GetSpacing()[0]**3
        # L2 = 1.
        L2 = 0.1

        # Initial values according to AHMOD in Chambolle2011
        # tau_n = 0.02
        # sigma_n = 4. / (L2 * tau_n)
        # gamma = 0.35 * self._lambda

        # Initial values according to ALG2 in Chambolle2011
        tau_n = 1. / np.sqrt(L2)
        sigma_n = 1. / (L2 * tau_n)
        gamma = 0.35 * self._lambda

        # Get data array of current volume estimate
        HR_nda_vec = sitk.GetArrayFromImage(self._HR_volume.sitk).flatten()
        x_mean = np.array(HR_nda_vec)
        p_n = self._update_dual_variable(0, sigma_n, x_mean)

        # Debug
        if debug:
            recons = []
            recons.insert(0, self._HR_volume.sitk)

        # Pre-compute static part of right hand-side
        b = np.zeros(self._N_total_slice_voxels + self._N_voxels_HR_volume)
        b[0:self._N_total_slice_voxels] = self._get_M_y()

        for iter in range(0, self._primal_dual_iterations):
            ph.print_subtitle("Primal-Dual Iteration %d/%d:" %
                              (iter+1, self._primal_dual_iterations))

            # Perform step for dual variable
            p_n = self._update_dual_variable(p_n, sigma_n, x_mean)

            # Perform step for primal variable
            HR_nda_vec_n = self._update_primal_variable(
                HR_nda_vec, p_n, tau_n, self._lambda, b)

            # Perform parameter updates
            # theta_n = 0.  # Arrow-Hurwicz Method
            theta_n = 1. / np.sqrt(1. + 2. * gamma * tau_n)
            tau_n *= theta_n
            sigma_n /= theta_n

            # Perform update mean variable
            x_mean = HR_nda_vec_n + theta_n * (HR_nda_vec_n - HR_nda_vec)

            if debug:
                HR_volume_sitk = sitk.GetImageFromArray(
                    HR_nda_vec_n.reshape(self._HR_shape_nda))
                HR_volume_sitk.CopyInformation(self._HR_volume.sitk)
                recons.insert(0, HR_volume_sitk)

                ph.killall_itksnap()
                sitkh.show_sitk_image(recons)

            # Prepare for next iteration
            HR_nda_vec = HR_nda_vec_n

        # Set elapsed time
        self._elapsed_time_sec = time.time() - time_start

        # Update volume
        self._HR_volume.itk = self._get_itk_image_from_array_vec(
            HR_nda_vec, self._HR_volume.itk)
        self._HR_volume.sitk = sitkh.get_sitk_from_itk_image(
            self._HR_volume.itk)

    def _print_info_text(self):
        ph.print_title("Primal-Dual Solver:")
        ph.print_debug_info("Chosen regularization type: %s" %
                            (self._reg_type))
        ph.print_debug_info(
            "Regularization parameter alpha: %g" % (self._alpha))
        if self._loss in ["huber"]:
            ph.print_debug_info("Loss function: %s (gamma = %g)" %
                                (self._loss, self._huber_gamma))
        else:
            ph.print_debug_info("Loss function: %s" % (self._loss))
        ph.print_debug_info("Number of Primal-Dual iterations: %d" %
                            (self._primal_dual_iterations))
        ph.print_debug_info("Minimizer: %s" % (self._minimizer))
        ph.print_debug_info(
            "Maximum number of iterations: %d" % (self._iter_max))

    def _update_dual_variable(self, p_n, sigma_n, x_mean):
        p_n = p_n + sigma_n * self._D(x_mean.reshape(self._HR_shape_nda))
        return p_n / np.maximum(1, np.abs(p_n))

    def _update_primal_variable(self, HR_nda_vec, p_n, tau_n, lmbda, b):

        A_fw = lambda x: self._get_A(x, tau_n * lmbda)
        A_bw = lambda x: self._get_A_adj(x, tau_n * lmbda)

        b[-self._N_voxels_HR_volume:] = HR_nda_vec - tau_n * self._D_adj(p_n)

        return self._get_approximate_solution[self._minimizer](
            A_fw, A_bw, b)

    def _get_A(self, x, tau_lambda):

        # Allocate memory
        A_x = np.zeros(self._N_total_slice_voxels + self._N_voxels_HR_volume)

        # Compute MAx
        A_x[0:-self._N_voxels_HR_volume] = tau_lambda * self._MA(x)

        # Add x
        A_x[-self._N_voxels_HR_volume:] = x

        return A_x

    def _get_A_adj(self, stacked_slices_nda_vec, tau_lambda):

        # Compute A'M y[upper]
        A_adj_y = tau_lambda * self._A_adj_M(stacked_slices_nda_vec)

        # Add y[lower]
        A_adj_y = A_adj_y + stacked_slices_nda_vec[-self._N_voxels_HR_volume:]

        return A_adj_y
