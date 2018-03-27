##
# \file robust_motion_estimator.py
# \brief      Class to estimate motion parameters from estimated
#             transformations.
#
# Regularisation of estimated motion parameters for robust slice-motion
# estimates
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Feb 2018
#

import os
import scipy
import pymc3
import theano
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh


##
# Class to compute robust slice-motion estimates
# \date       2018-03-26 16:47:47-0600
#
class RobustMotionEstimator(object):

    def __init__(self, transforms_sitk, interleave=2):
        self._transforms_sitk = transforms_sitk
        self._interleave = interleave

        self._robust_transforms_sitk = [None] * len(self._transforms_sitk)

    ##
    # Gets the robust transforms sitk.
    # \date       2018-03-26 16:46:00-0600
    #
    # \param      self  The object
    #
    # \return     The robust transforms sitk as list of sitk.Transforms
    #
    def get_robust_transforms_sitk(self):
        robust_transforms_sitk = [
            sitkh.copy_transform_sitk(t) for t in self._robust_transforms_sitk]
        return robust_transforms_sitk

    ##
    # Run Gaussian process smoothing for each dof individually
    # \date       2018-03-13 19:31:02+0000
    # \see http://docs.pymc.io/notebooks/GP-smoothing.html
    #
    # \param      self  The object
    #
    def run_gaussian_process_smoothing(self, smoothing=0.5):

        params_nda = self._get_transformation_params_nda(self._transforms_sitk)

        # Iterate over each dof
        for i_dof in range(params_nda.shape[0]):

            # Smooth each interleave package separately
            for i in range(self._interleave):
                indices = np.arange(i, params_nda.shape[1], self._interleave)
                y = params_nda[i_dof, indices]
                y_smoothed = self._run_gaussian_process_smoothing(
                    y=y, smoothing=smoothing)
                params_nda[i_dof, indices] = y_smoothed

        self._update_robust_transforms_sitk_from_parameters(params_nda)

    def _run_gaussian_process_smoothing(self, y, smoothing):
        LARGE_NUMBER = 1e5
        model = pymc3.Model()
        with model:
            smoothing_param = theano.shared(smoothing)
            mu = pymc3.Normal("mu", sd=LARGE_NUMBER)
            tau = pymc3.Exponential("tau", 1.0 / LARGE_NUMBER)
            z = pymc3.distributions.timeseries.GaussianRandomWalk(
                "z",
                mu=mu,
                tau=tau / (1.0 - smoothing_param),
                shape=y.shape,
            )
            obs = pymc3.Normal(
                "obs",
                mu=z,
                tau=tau / smoothing_param,
                observed=y,
            )
            res = pymc3.find_MAP(
                vars=[z], fmin=scipy.optimize.fmin_l_bfgs_b)
            return res['z']

    # ##
    # # { function_description }
    # # \date       2018-03-13 19:23:39+0000
    # # \see http://docs.pymc.io/notebooks/GP-slice-sampling.html
    # #
    # # \param      self  The object
    # #
    # # \return     { description_of_the_return_value }
    # #
    # def run_gaussian_process_regression(self):
    #     params_nda = self._get_transformation_params_nda(self._transforms_sitk)

    #     # number of training points
    #     n = params_nda.shape[1]
    #     X0 = np.arange(0, params_nda.shape[1])[:, None]

    #     # Number of points at which to interpolate
    #     X = np.arange(0, params_nda.shape[1])[:, None]

    #     # Covariance kernel parameters
    #     noise = 0.1
    #     lengthscale = 0.3
    #     f_scale = 1

    #     cov = f_scale * pymc3.gp.cov.ExpQuad(1, lengthscale)

    #     K = cov(X0)
    #     K_s = cov(X0, X)
    #     K_noise = K + noise * theano.tensor.eye(n)

    #     # Add very slight perturbation to the covariance matrix diagonal to
    #     # improve numerical stability
    #     K_stable = K + 1e-12 * theano.tensor.eye(n)

    #     # Observed data
    #     f = np.random.multivariate_normal(mean=np.zeros(n), cov=K_noise.eval())

    #     fig, ax = plt.subplots(figsize=(14, 6))
    #     ax.scatter(X0, f, s=40, color='b', label='True points')
    #     ax.set_xticks(X0)

    #     # Analytically compute posterior mean
    #     L = np.linalg.cholesky(K_noise.eval())
    #     alpha = np.linalg.solve(L.T, np.linalg.solve(L, f))
    #     post_mean = np.dot(K_s.T.eval(), alpha)

    #     ax.plot(X, post_mean, color='g', alpha=0.8, label='Posterior mean')

    #     ax.legend()

    #     plt.show(True)

    ##
    # Shows the estimated transform parameters.
    # \date       2018-03-26 16:45:27-0600
    #
    # \param      self        The object
    # \param      title       The title
    # \param      fullscreen  The fullscreen
    #
    def show_estimated_transform_parameters(
            self, dir_output=None, title="RobustMotionEstimator", fullscreen=1):
        params_nda = self._get_transformation_params_nda(self._transforms_sitk)
        robust_params_nda = self._get_transformation_params_nda(
            self.get_robust_transforms_sitk())

        dof = params_nda.shape[0]

        N_rows = np.ceil(dof / 2.)
        i_ref_marker = 0

        fig = plt.figure(title)
        fig.clf()
        for i_dof in range(dof):
            x = np.arange(params_nda.shape[1])
            y1 = params_nda[i_dof, :]
            y2 = robust_params_nda[i_dof, :]

            ax = plt.subplot(N_rows, 2, i_dof + 1)
            ax.plot(x, y1,
                    marker=ph.MARKERS[i_ref_marker],
                    color=ph.COLORS_TABLEAU20[0],
                    linestyle=":",
                    label="original",
                    markerfacecolor="w",
                    )
            ax.plot(x, y2,
                    marker=ph.MARKERS[i_ref_marker],
                    color=ph.COLORS_TABLEAU20[2],
                    linestyle="-.",
                    label="robust",
                    )
            ax.set_xticks(x)
            plt.ylabel(sitkh.TRANSFORM_SITK_DOF_LABELS_LONG[dof][i_dof])
        plt.legend(loc="best")
        plt.xlabel('Slice')
        plt.suptitle(title)

        if fullscreen:
            try:
                # Open windows (and also save them) in full screen
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()
            except:
                pass

        plt.show(block=False)

        if dir_output is not None:
            filename = "%s.pdf" % title
            ph.save_fig(fig, dir_output, filename)

    ##
    # Get transformation parameters from sitk transform
    # \date       2018-03-26 16:43:20-0600
    #
    # \param      self             The object
    # \param      transforms_sitk  List of sitk.Transforms
    #
    # \return     The transformation parameters as (dof x #slices)-numpy array
    #
    def _get_transformation_params_nda(self, transforms_sitk):
        dof = len(transforms_sitk[0].GetParameters())
        N_transformations = len(transforms_sitk)

        params_nda = np.zeros((dof, N_transformations))
        for i, transform_sitk in enumerate(transforms_sitk):
            params_nda[:, i] = np.array(transform_sitk.GetParameters())

        # params_nda = self._apply_interleave(params_nda)

        return params_nda

    ##
    # Update robust transformations given parameter estimates
    # \date       2018-03-26 15:52:19-0600
    #
    # \param      self        The object
    # \param      params_nda  The parameters as (dof x #slices)-numpy array
    #
    def _update_robust_transforms_sitk_from_parameters(self, params_nda):

        # params_nda = self._undo_interleave(params_nda)

        for i, transform_sitk in enumerate(self._transforms_sitk):
            robust_transforms_sitk = sitkh.copy_transform_sitk(transform_sitk)
            robust_transforms_sitk.SetParameters(params_nda[:, i])

            self._robust_transforms_sitk[i] = robust_transforms_sitk

    # def _apply_interleave(self, params_nda):

    #     indices = []
    #     for i in range(self._interleave):
    #         indices.append(np.arange(i, self._interleave, params_nda.shape[1]))

    # def _undo_interleave(self, params_nda):
