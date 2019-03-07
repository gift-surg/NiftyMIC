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

    def __init__(self, transforms_sitk, interleave, verbose=True):
        self._transforms_sitk = transforms_sitk
        self._interleave = interleave
        self._verbose = verbose

        self._robust_transforms_sitk = {}

    ##
    # Gets the robust transforms sitk.
    # \date       2018-03-26 16:46:00-0600
    #
    # \param      self  The object
    #
    # \return     The robust transforms sitk as list of sitk.Transforms
    #
    def get_robust_transforms_sitk(self):
        return self._robust_transforms_sitk

    def run(self, parameter):
        slice_indices, params = self._get_transformation_params_nda(
            self._transforms_sitk)

        temporal_packages = self._get_temporal_packages(slice_indices)
        # print(slice_indices)
        # for t in temporal_packages:
        #     print(t)

        for dof in range(params.shape[0]):
            if self._verbose:
                ph.print_info("DOF %d/%d ... " % (
                    dof + 1, params.shape[0]))
            for package in temporal_packages:

                # continue in case no slices in subpackage left
                if len(package.keys()) == 0:
                    continue

                t = sorted(package.keys())
                slices_package = [slice_indices.index(
                    package[t_i]) for t_i in t]
                y = params[dof, slices_package]

                y_est = self._run_gaussian_process_smoothing(
                    t, y, smoothing=parameter)
                # y_est = self._run_robust_gaussian_process_regression_map(t, y)

                params[dof, slices_package] = y_est

        self._update_robust_transforms_sitk_from_parameters(params)

    @staticmethod
    def _run_gaussian_process_smoothing(x, y, smoothing):

        # LARGE_NUMBER = 1000000
        with pymc3.Model() as model:
            smoothing_param = theano.shared(smoothing)
            # mu = pymc3.Normal("mu", sd=LARGE_NUMBER)
            # tau = pymc3.Exponential("tau", 1.0 / LARGE_NUMBER)
            # z = pymc3.distributions.timeseries.GaussianRandomWalk(
            #     "z",
            #     mu=mu,
            #     tau=tau / (1.0 - smoothing_param),
            #     shape=y.shape,
            # )
            # obs = pymc3.Normal(
            #     "obs",
            #     mu=z,
            #     tau=tau / smoothing_param,
            #     observed=y,
            # )

            sd = pymc3.Exponential("sd", 1)
            z = pymc3.distributions.timeseries.GaussianRandomWalk(
                "z",
                mu=0,
                sd=(1.0 - smoothing_param) * sd,
                shape=y.shape,
            )

            nu = pymc3.Gamma("nu", alpha=2, beta=1)
            obs = pymc3.StudentT(
                "obs",
                mu=z,
                sd=sd * smoothing_param,
                nu=nu,
                observed=y,
            )

            res = pymc3.find_MAP(vars=[z], method="L-BFGS-B")
            return res['z']

    @staticmethod
    def _run_robust_gaussian_process_regression_map(x, y):
        x = np.array(x)
        with pymc3.Model() as model:
            ell = pymc3.Gamma("ell", alpha=2, beta=1)
            eta = pymc3.HalfCauchy("eta", beta=5)

            cov = eta**2 * pymc3.gp.cov.Matern52(1, ell)
            gp = pymc3.gp.Latent(cov_func=cov)

            f = gp.prior("f", X=x.reshape(-1, 1))

            sigma = pymc3.HalfCauchy("sigma", beta=2)
            # sigma = pymc3.Normal("sigma")
            # sigma = 0.1
            nu = pymc3.Gamma("nu", alpha=2, beta=1)
            # sigma = 0.01
            # nu = 0.01
            # y_ = pymc3.StudentT("y", mu=f, lam=1.0/sigma, nu=nu, observed=y)
            y_ = pymc3.StudentT("y", mu=f, sd=sigma, nu=nu, observed=y)

            # res = pymc3.find_MAP(model=model, method="L-BFGS-B")
            res = pymc3.find_MAP(vars=[f], method="L-BFGS-B")
            return res["f"]

    # ##
    # # Run Gaussian process smoothing for each dof individually
    # # \date       2018-03-13 19:31:02+0000
    # # \see http://docs.pymc.io/notebooks/GP-smoothing.html
    # #
    # # \param      self  The object
    # #
    # def run_gaussian_process_smoothing(self, smoothing=0.5):

    #     params_nda = self._get_transformation_params_nda(self._transforms_sitk)

    #     # Iterate over each dof
    #     for i_dof in range(params_nda.shape[0]):

    #         # Smooth each interleave package separately
    #         for i in range(self._interleave):
    #             indices = np.arange(i, params_nda.shape[1], self._interleave)
    #             y = params_nda[i_dof, indices]
    #             y_smoothed = self._run_gaussian_process_smoothing(
    #                 y=y, smoothing=smoothing)
    #             params_nda[i_dof, indices] = y_smoothed

    #     self._update_robust_transforms_sitk_from_parameters(params_nda)

    # def _run_gaussian_process_smoothing(self, y, smoothing):
    #     LARGE_NUMBER = 1e5
    #     model = pymc3.Model()
    #     with model:
    #         smoothing_param = theano.shared(smoothing)
    #         mu = pymc3.Normal("mu", sd=LARGE_NUMBER)
    #         tau = pymc3.Exponential("tau", 1.0 / LARGE_NUMBER)
    #         z = pymc3.distributions.timeseries.GaussianRandomWalk(
    #             "z",
    #             mu=mu,
    #             tau=tau / (1.0 - smoothing_param),
    #             shape=y.shape,
    #         )
    #         obs = pymc3.Normal(
    #             "obs",
    #             mu=z,
    #             tau=tau / smoothing_param,
    #             observed=y,
    #         )
    #         res = pymc3.find_MAP(vars=[z], method="L-BFGS-B")
    #         return res['z']

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
        self,
        path_to_file=None,
        title="RobustMotionEstimator",
        fullscreen=1,
    ):
        indices, params_nda = self._get_transformation_params_nda(
            self._transforms_sitk)
        robust_params_nda = self._get_transformation_params_nda(
            self.get_robust_transforms_sitk())[1]

        dof = params_nda.shape[0]

        N_rows = np.ceil(dof / 2.)
        i_ref_marker = 0

        subpackages = self._get_temporal_packages(indices)

        fig = plt.figure(title, figsize=(15, 10))
        for i_dof in range(dof):
            y1 = params_nda[i_dof, :]
            y2 = robust_params_nda[i_dof, :]

            ax = plt.subplot(N_rows, 2, i_dof + 1)
            ax.plot(indices, y1,
                    marker=ph.MARKERS[i_ref_marker],
                    color=ph.COLORS_TABLEAU20[0],
                    linestyle="",
                    # linestyle=":",
                    label="original",
                    markerfacecolor="w",
                    )

            # print connecting line between subpackage slices
            ls = ["--", ":", "-."]
            for i_p, p in enumerate(subpackages):
                t = sorted(p.keys())

                slices_package = [indices.index(p[t_i]) for t_i in t]
                y = y1[slices_package]
                x = [indices[i] for i in slices_package]
                ax.plot(x, y,
                        marker=".",
                        # color=ph.COLORS_TABLEAU20[2 + i_p],
                        color=[0.7, 0.7, 0.7],
                        linestyle=ls[i_p],
                        )
            for i in range(len(y1)):
                ax.plot([indices[i], indices[i]], [y1[i], y2[i]],
                        linestyle="-",
                        marker="",
                        color=ph.COLORS_TABLEAU20[2],
                        )
            ax.plot(indices, y2,
                    marker=ph.MARKERS[i_ref_marker],
                    color=ph.COLORS_TABLEAU20[2],
                    # linestyle="-",
                    linestyle="",
                    label="robust",
                    )
            ax.set_xticks(indices)
            plt.ylabel(sitkh.TRANSFORM_SITK_DOF_LABELS_LONG[dof][i_dof])
        plt.legend(loc="best")
        plt.xlabel('Slice')
        plt.suptitle(title)

        plt.show(block=False)

        if path_to_file is not None:
            ph.create_directory(os.path.dirname(path_to_file))
            fig.savefig(path_to_file)
            ph.print_info("Figure written to %s" % path_to_file)
        plt.close()

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
        slice_indices = sorted(transforms_sitk.keys())
        dof = len(transforms_sitk[slice_indices[0]].GetParameters())
        N_transformations = len(slice_indices)

        params_nda = np.zeros((dof, N_transformations))
        for i, index in enumerate(slice_indices):
            params_nda[:, i] = np.array(transforms_sitk[index].GetParameters())

        # params_nda = self._get_temporal_packages(params_nda)

        return slice_indices, params_nda

    ##
    # Update robust transformations given parameter estimates
    # \date       2018-03-26 15:52:19-0600
    #
    # \param      self        The object
    # \param      params_nda  The parameters as (dof x #slices)-numpy array
    #
    def _update_robust_transforms_sitk_from_parameters(self, params_nda):

        self._robust_transforms_sitk = {}
        slice_indices = sorted(self._transforms_sitk.keys())
        for i, slice_index in enumerate(slice_indices):
            t_sitk = self._transforms_sitk[slice_index]
            robust_t_sitk = sitkh.copy_transform_sitk(t_sitk)
            robust_t_sitk.SetParameters(params_nda[:, i])

            self._robust_transforms_sitk[slice_index] = robust_t_sitk

    def _get_temporal_packages(self, slice_indices):

        slice_acquisitions = np.arange(
            np.array(slice_indices).min(), np.array(slice_indices).max() + 1)

        n_slice = np.array(slice_indices).max()
        packages = []
        for i in range(self._interleave):
            orig_acquisitions = np.arange(
                i, n_slice + 1, self._interleave)

            dic = {
                t: index
                for (t, index) in enumerate(orig_acquisitions)
                if index in slice_indices
            }
            packages.append(dic)

        return packages
