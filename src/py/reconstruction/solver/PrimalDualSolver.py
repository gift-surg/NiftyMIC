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
                 reg_type="TV",
                 reg_huber_gamma=0.05,
                 primal_dual_iterations=10,
                 primal_dual_parameter_method="ALG2",
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

        # regularization type
        self._reg_type = reg_type

        # number of primal-dual iterations
        self._primal_dual_iterations = primal_dual_iterations

        # parameter used for Huber regularizer
        self._reg_huber_gamma = reg_huber_gamma

        # update step for dual variable depends on the chosen regularizer
        self._get_update_dual_variable = {
            "TV": self._get_update_dual_variable_tv,
            "huber": self._get_update_dual_variable_huber,
        }

        # define method to update parameter
        self._primal_dual_parameter_method = primal_dual_parameter_method

        # parameter initialization depend on chosen method
        self._get_initial_tau_sigma = {
            "ALG2": self._get_initial_tau_sigma_alg2,
            "AHMOD": self._get_initial_tau_sigma_ahmod,
        }

        # parameter updates depend on chosen method
        self._get_update_theta_tau_sigma = {
            "ALG2": self._get_update_theta_tau_sigma_alg2,
            "AHMOD": self._get_update_theta_tau_sigma_ahmod,
        }

        self._get_update_primal_variable = {
            "linear": self._get_update_primal_variable_linear,
        }

    def get_setting_specific_filename(self, prefix="SRR_"):

        # Build filename
        filename = prefix
        filename += "stacks" + str(len(self._stacks))
        if self._alpha > 0:
            filename += "_" + self._reg_type
            if self._reg_type == "huber":
                filename += "_gamma" + str(self._reg_huber_gamma)
            filename += "_PrimalDual"
            # filename += "_" + self._primal_dual_parameter_method
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

    def run_reconstruction(self, verbose=0):

        self._print_info_text()

        # regularization parameter lambda as used in Chambolle2011
        lmbda = 1. / self._alpha

        # L^2 = ||K||^2 = ||\nabla||^2 = ||div||^2 <= 16/h^2 in 3D
        # However, it seems that the smaller L2 the bigger the effect of TV
        # regularization. Try, e.g. L2 = 1.
        L2 = 16. / self._HR_volume.sitk.GetSpacing()[0]**2

        time_start = time.time()

        # In case either dual or primal objectives is uniformly convex
        HR_nda_vec = self._get_reconstruction_alg2(lmbda=lmbda,
                                                   L2=L2,
                                                   verbose=verbose)

        # In case both dual and primal objectives are uniformly convex
        # HR_nda_vec = self._get_reconstruction_alg3(lmbda=lmbda,
        #                                            L2=L2,
        #                                            verbose=verbose)

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
                            (self._reg_type), newline=False)
        if self._reg_type == "huber":
            print(" (gamma = %g)" % (self._reg_huber_gamma))
        else:
            print("")
        ph.print_debug_info("Strategy for parameter update: %s"
                            % (self._primal_dual_parameter_method))
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

    ##
    # Get reconstruction based on Alg. 2, p. 122 in Chambolle2011
    #
    # Convergence O(1/N^2) in case either primal or dual objective, i.e. G or
    # F^*, is uniformly convex.
    # \date       2017-07-18 23:13:44+0100
    #
    # \param      self     The object
    # \param      lmbda    Regularization parameter
    # \param      L2       Squared operator norm
    # \param      verbose  verbose output. True/False
    #
    # \return     data array of obtained reconstruction.
    #
    def _get_reconstruction_alg2(self, lmbda, L2, verbose):

        # Dynamic step sizes for primal and dual variable, see p.127
        tau_n, sigma_n, gamma = self._get_initial_tau_sigma[
            self._primal_dual_parameter_method](L2=L2, lmbda=lmbda)

        # Get data array of current volume estimate
        HR_nda_vec = sitk.GetArrayFromImage(self._HR_volume.sitk).flatten()
        x_mean = np.array(HR_nda_vec)
        p_n = self._get_update_dual_variable[
            self._reg_type](p_n=0, sigma_n=sigma_n, x_mean=x_mean)

        if verbose:
            recons = []
            recons.insert(0, self._HR_volume.sitk)

        # Pre-compute static part of right hand-side
        b = np.zeros(self._N_total_slice_voxels + self._N_voxels_HR_volume)
        b[0:self._N_total_slice_voxels] = self._get_M_y()

        for iter in range(0, self._primal_dual_iterations):
            ph.print_subtitle("Primal-Dual Iteration %d/%d:" %
                              (iter+1, self._primal_dual_iterations))
            if iter > 0:
                ph.print_debug_info("theta_n = %g" % (theta_n))
            ph.print_debug_info("tau_n = %g" % (tau_n))
            ph.print_debug_info("sigma_n = %g" % (sigma_n))

            # Perform step for dual variable
            p_n = self._get_update_dual_variable[self._reg_type](
                p_n=p_n, sigma_n=sigma_n, x_mean=x_mean)

            # Perform step for primal variable
            HR_nda_vec_n = self._get_update_primal_variable[self._loss](
                x_n=HR_nda_vec, p_n=p_n, tau_n=tau_n, lmbda=lmbda, b=b)

            # Perform parameter updates
            theta_n, tau_n, sigma_n = self._get_update_theta_tau_sigma[
                self._primal_dual_parameter_method](L2, gamma, tau_n, sigma_n)

            # Perform update mean variable
            x_mean = HR_nda_vec_n + theta_n * (HR_nda_vec_n - HR_nda_vec)

            if verbose:
                HR_volume_sitk = sitk.GetImageFromArray(
                    HR_nda_vec_n.reshape(self._HR_shape_nda))
                HR_volume_sitk.CopyInformation(self._HR_volume.sitk)
                recons.insert(0, HR_volume_sitk)

                ph.killall_itksnap()
                sitkh.show_sitk_image(recons)

            # Prepare for next iteration
            HR_nda_vec = HR_nda_vec_n

        return HR_nda_vec

    ##
    # Get reconstruction based on Alg. 3, p. 131 in Chambolle2011
    #
    # Convergence O(w^{N/2}) in case both primal and dual objective are
    # uniformly convex.
    # \date       2017-07-18 23:13:44+0100
    #
    # \param      self     The object
    # \param      lmbda    Regularization parameter
    # \param      L2       Squared operator norm
    # \param      verbose  verbose output. True/False
    #
    # \return     data array of obtained reconstruction.
    #
    def _get_reconstruction_alg3(self, lmbda, L2, verbose):

        #
        gamma = lmbda

        #
        delta = self._reg_huber_gamma

        #
        mu = 2. * np.sqrt(gamma * delta / L2)

        # relaxation parameter in [1/(1+mu), 1]
        theta = 1. / (1. + mu)

        # step size dual variable
        sigma = mu / (2. * delta)

        # step size primal variable
        tau = mu / (2. * gamma)

        # Get data array of current volume estimate
        HR_nda_vec = sitk.GetArrayFromImage(self._HR_volume.sitk).flatten()
        x_mean = np.array(HR_nda_vec)
        p_n = self._get_update_dual_variable[
            self._reg_type](p_n=0, sigma_n=sigma, x_mean=x_mean)

        if verbose:
            recons = []
            recons.insert(0, self._HR_volume.sitk)

        # Pre-compute static part of right hand-side
        b = np.zeros(self._N_total_slice_voxels + self._N_voxels_HR_volume)
        b[0:self._N_total_slice_voxels] = self._get_M_y()

        for iter in range(0, self._primal_dual_iterations):
            ph.print_subtitle("Primal-Dual Iteration %d/%d:" %
                              (iter+1, self._primal_dual_iterations))

            # Perform step for dual variable
            p_n = self._get_update_dual_variable[self._reg_type](
                p_n=p_n, sigma_n=sigma, x_mean=x_mean)

            # Perform step for primal variable
            HR_nda_vec_n = self._get_update_primal_variable[self._loss](
                x_n=HR_nda_vec, p_n=p_n, tau_n=tau, lmbda=lmbda, b=b)

            # Perform update mean variable
            x_mean = HR_nda_vec_n + theta * (HR_nda_vec_n - HR_nda_vec)

            if verbose:
                HR_volume_sitk = sitk.GetImageFromArray(
                    HR_nda_vec_n.reshape(self._HR_shape_nda))
                HR_volume_sitk.CopyInformation(self._HR_volume.sitk)
                recons.insert(0, HR_volume_sitk)

                ph.killall_itksnap()
                sitkh.show_sitk_image(recons)

            # Prepare for next iteration
            HR_nda_vec = HR_nda_vec_n

        return HR_nda_vec

    ##
    # Gets the initial step sizes tau_0, sigma_0 and the Lipschitz parameter
    # gamma according to ALG2 method in Chambolle2011, p.133
    #
    # tau_0 and sigma_0 such that tau_0 * sigma_0 * L^2 = 1
    # \date       2017-07-18 17:57:33+0100
    #
    # \param      self   The object
    # \param      L2     Squared operator norm
    # \param      lmbda  Regularization parameter
    #
    # \return     tau0, sigma0, gamma
    #
    def _get_initial_tau_sigma_alg2(self, L2, lmbda):
        # Initial values according to ALG2 in Chambolle2011
        tau0 = 1. / np.sqrt(L2)
        sigma0 = 1. / (L2 * tau0)
        gamma = 0.35 * lmbda
        return tau0, sigma0, gamma

    ##
    # Gets the update of the variable relaxation parameter
    # \f$\theta_n\in[0,1]\f$ and the dynamic step sizes
    # \f$\tau_n,\,\sigma_n>0\f$ for the primal and dual variable, respectively.
    #
    # Update is performed according to ALG2 in Chambolle2011, p.133. It always
    # holds tau_n * sigma_n * L^2 = 1.
    # \date       2017-07-18 18:16:28+0100
    #
    # \param      self     The object
    # \param      L2       Squared operator norm
    # \param      gamma    Lipschitz parameter
    # \param      tau_n    Dynamic step size for primal variable
    # \param      sigma_n  Dynamic step size for dual variable
    #
    # \return     theta_n, tau_n, sigma_n update
    #
    def _get_update_theta_tau_sigma_alg2(self, L2, gamma, tau_n, sigma_n):
        theta_n = 1. / np.sqrt(1. + 2. * gamma * tau_n)
        tau_n = tau_n * theta_n
        sigma_n = sigma_n / theta_n
        return theta_n, tau_n, sigma_n

    ##
    # Gets the initial step sizes tau_0, sigma_0 and the Lipschitz parameter
    # gamma according to AHMOD, i.e. Arrow-Hurwicz method, in Chambolle2011,
    # p.133
    #
    # tau_0 and sigma_0 such that tau_0 * sigma_0 * L^2 = 4
    # \date       2017-07-18 17:56:36+0100
    #
    # \param      self   The object
    # \param      L2     Squared operator norm
    # \param      lmbda  Regularization parameter
    #
    # \return     tau0, sigma0, gamma
    #
    def _get_initial_tau_sigma_ahmod(self, L2, lmbda):
        # Initial values according to AHMOD in Chambolle2011
        tau0 = 0.02
        sigma0 = 4. / (L2 * tau0)
        gamma = 0.35 * lmbda
        return tau0, sigma0, gamma

    ##
    # Gets the update of the variable relaxation parameter
    # \f$\theta_n\in[0,1]\f$ and the dynamic step sizes
    # \f$\tau_n,\,\sigma_n>0\f$ for the primal and dual variable, respectively.
    #
    # Update is performed according to AHMOD, i.e. Arrow-Hurwicz method, in
    # Chambolle2011, p.133. It always holds tau_n * sigma_n * L^2 = 4.
    # \date       2017-07-18 18:16:28+0100
    #
    # \param      self     The object
    # \param      L2       Squared operator norm
    # \param      gamma    Lipschitz parameter
    # \param      tau_n    Dynamic step size for primal variable
    # \param      sigma_n  Dynamic step size for dual variable
    #
    # \return     theta_n, tau_n, sigma_n update
    #
    def _get_update_theta_tau_sigma_ahmod(self, L2, gamma, tau_n, sigma_n):
        theta_n = 1. / np.sqrt(1. + 2. * gamma * tau_n)
        tau_n = tau_n * theta_n
        sigma_n = sigma_n / theta_n
        return 0., tau_n, sigma_n

    ##
    # Gets the update of the dual variable in case of TV regularization
    #
    # Given the regularizer \f$ F(K\vec{x}) = \Vert K\vec{x} \Vert_{\ell^1}\f$
    # compute the update of the dual variable
    # \f$ \vec{p}_{n+1} := \text{prox}_{\sigma_n F^*}(\widetilde{\vec{p}}_n) =
    # \frac{\widetilde{\vec{p}}_n}{\max\{1,\,|\widetilde{\vec{p}}_n|\}}
    # \f$ with
    # \f$ \widetilde{\vec{p}}_n := \vec{p}_n + \sigma_n K \overline{\vec{x}}_n
    # \f$
    # \date       2017-07-18 20:04:50+0100
    #
    # \param      self     The object
    # \param      p_n      Dual variable as 3 * N_voxels_HR_volume numpy array
    # \param      sigma_n  Dynamic step size for dual variable, scalar > 0
    # \param      x_mean   x_mean as linear combination from previous primal
    #                      variable as N_voxels_HR_volume array
    #
    # \return     Update of dual variable as 3 * N_voxels_HR_volume numpy
    #             array.
    #
    def _get_update_dual_variable_tv(self, p_n, sigma_n, x_mean):

        # compute p_n + sigma_n * K x_mean
        p_n = p_n + sigma_n * self._D(x_mean.reshape(self._HR_shape_nda))

        # apply proximal map for TV
        return p_n / np.maximum(1, np.abs(p_n))

    ##
    # Gets the update of the dual variable in case of Huber regularization.
    #
    # Given the regularizer
    # \f$ F(K\vec{x}) = |K\vec{x}|_\gamma
    # \f$ with the Huber function
    # \f[ |x|_\gamma := \begin{cases} \frac{|x|^2}{2\gamma}, & |x| \le \gamma
    # \\ |x| - \gamma/2, & |x| > \gamma \end{cases}
    # \f] compute the update of the dual variable
    # \f$ \vec{p}_{n+1} := \text{prox}_{\sigma_n F^*}(\widetilde{\vec{p}}_n) =
    # \frac{\widetilde{\vec{p}}_n}{\max\{1,\,|\widetilde{\vec{p}}_n|\}}
    # \f$ with
    # \f$ \widetilde{\vec{p}}_n := \frac{\vec{p}_n + \sigma_n K
    # \overline{\vec{x}}_n}{1+\sigma_n\gamma}
    # \f$
    # \date       2017-07-18 21:43:07+0100
    #
    # \param      self     The object
    # \param      p_n      Dual variable as 3 * N_voxels_HR_volume numpy array
    # \param      sigma_n  Dynamic step size for dual variable, scalar > 0
    # \param      x_mean   x_mean as linear combination from previous primal
    #                      variable as N_voxels_HR_volume array
    #
    # \return     Update of dual variable as 3 * N_voxels_HR_volume numpy
    #             array.
    #
    def _get_update_dual_variable_huber(self, p_n, sigma_n, x_mean):

        # compute p_n + sigma_n * K x_mean
        p_n = p_n + sigma_n * self._D(x_mean.reshape(self._HR_shape_nda))

        # apply proximal map for Huber
        p_n /= (1. + sigma_n * self._reg_huber_gamma)
        return p_n / np.maximum(1, np.abs(p_n))

    ##
    # Gets the update of the primal variable.
    #
    # Solve the minimization problem given by the proximal operator, i.e.
    # \f[ \vec{x}_{n+1} := \text{prox}_{\tau_n G}(\vec{x}_n - \tau_n K^*
    # \vec{p}_{n+1}) = \text{argmin}_{\vec{x}} \Big[ \frac{1}{2}\Vert \vec{x} -
    # (\vec{x}_n - \tau_n K^* \vec{p}_{n+1})\Vert_{\ell^2}^2 + \tau_n
    # G(\vec{x}) \Big]\f] In case of
    # \f$G(\vec{x}) = \sum_{k=1}^K \frac{1}{2} \Vert \vec{y}_k - A_k \vec{x}
    # \Vert_{\ell^2}^2
    # \f$ this can be rephrased as linear least squares problem
    # \f$A \vec{x} = \vec{b}\f$ with
    # \f$A = \begin{pmatrix} \sqrt{\tau_n \lambda} A_1 \\ \sqrt{\tau_n
    # \lambda} A_2 \\ \vdots \\ \sqrt{\tau_n \lambda} A_K \\ I \end{pmatrix}
    # \f$ and
    # \f$ \vec{b} = \begin{pmatrix} \vec{y}_1 \\ \vec{y}_2 \\ \vdots \\
    # \vec{y}_K \\ \vec{x}_n - \tau_n K^* \vec{p}_{n+1}  \end{pmatrix} \f$
    # \date       2017-07-18 21:52:58+0100
    #
    # \param      self        The object
    # \param      x_n  HR 3D volume data flattened as numpy array
    # \param      p_n         Dual variable as 3 * N_voxels_HR_volume numpy
    #                         array
    # \param      tau_n       Dynamic step size for primal variable, scalar > 0
    # \param      lmbda       Regularization parameter
    # \param      b           Right hand-side as N_total_slice_voxels +
    #                         N_voxels_HR_volume numpy array
    #
    # \return     The update primal variable.
    #
    def _get_update_primal_variable_linear(self, x_n, p_n, tau_n, lmbda, b):

        sqrt_tau_lambda = np.sqrt(tau_n * lmbda)

        # Get forward and backward linear operators
        A_fw = lambda x: self._get_A(x, sqrt_tau_lambda)
        A_bw = lambda x: self._get_A_adj(x, sqrt_tau_lambda)

        # Update lower part of right hand-side vector
        b[-self._N_voxels_HR_volume:] = x_n - tau_n * self._D_adj(p_n)

        # Solve linear least squares problem
        x_n = self._get_approximate_solution[self._minimizer](A_fw, A_bw, b)

        return x_n

    ##
    # Get linear operator A to build linear least squares problem
    # \date       2017-07-18 22:14:16+0100
    #
    # \param      self             The object
    # \param      x                variable for optimization, dimension of HR
    #                              3D volume flattened as numpy array
    # \param      sqrt_tau_lambda  sqrt(tau_n * lambda)
    #
    # \return     Return Ax
    #
    def _get_A(self, x, sqrt_tau_lambda):

        # Allocate memory
        A_x = np.zeros(self._N_total_slice_voxels + self._N_voxels_HR_volume)

        # Compute MAx
        A_x[0:-self._N_voxels_HR_volume] = sqrt_tau_lambda * self._MA(x)

        # Add x
        A_x[-self._N_voxels_HR_volume:] = x

        # Return [ sqrt(tau * lambda) * MAx \\ x]
        return A_x

    ##
    # Get adjoint linear operator A^* to build linear least squares problem
    # \date       2017-07-18 22:14:16+0100
    #
    # \param      self             The object
    # \param      y                1D numpy array
    # \param      sqrt_tau_lambda  sqrt(tau_n * lambda)
    #
    # \return     Return A_adj y as N_voxels_HR_volume numpy array
    #
    def _get_A_adj(self, y, sqrt_tau_lambda):

        # Compute A'M y[upper]
        A_adj_y = sqrt_tau_lambda * self._A_adj_M(y)

        # Add y[lower]
        A_adj_y = A_adj_y + y[-self._N_voxels_HR_volume:]

        # Return [sqrt(tau * lambda) * A'M | I ] [y[upper] \\ y[lower]]
        return A_adj_y
